import glob
import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import v2 as T
from torchvision.transforms.functional import to_tensor


class BrailleDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, patch_size=224, transform_prob=0.1, train_val_ratio=0.8, is_train=True):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.sigma = 3
        self.transform_prob = transform_prob
        self.is_train = is_train

        all_paths = sorted(glob.glob(os.path.join(root_dir, "*.png")))
        split_idx = int(len(all_paths) * train_val_ratio)
        self.image_paths = all_paths[:split_idx] if is_train else all_paths[split_idx:]

        # light augmentation for grayscale tensor images using torchvision v2
        self.color_aug = T.Compose([
            T.RandomApply([
                T.ColorJitter(brightness=0.5, hue=0.3),
                T.RandomInvert(),
                T.RandomPosterize(bits=2),
                T.RandomSolarize(threshold=192.0 / 255.0),
                T.GaussianBlur(kernel_size=(3, 3))
            ], p=self.transform_prob)
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("L")
        w, h = img.size

        csv_path = os.path.splitext(img_path)[0] + ".csv"
        coords = pd.read_csv(csv_path).to_numpy().astype(np.float32)
        img_tensor = to_tensor(img)  # shape: [1, H, W]

        if self.is_train:
            if random.random() < 0.5:
                img_tensor = torch.flip(img_tensor, dims=[2])  # Horizontal flip
                coords[:, 0] = w - coords[:, 0]

            if random.random() < 0.5:
                img_tensor = torch.flip(img_tensor, dims=[1])  # Vertical flip
                coords[:, 1] = h - coords[:, 1]

            img_tensor, coords = self.apply_random_zoom(img_tensor, coords, zoom_range=(0.8, 1.2))
            img_tensor, coords = self.apply_random_crop(img_tensor, coords, crop_size=self.patch_size)

            angle = random.uniform(-45, 45)
            translate = (random.uniform(-0.1, 0.1) * w, random.uniform(-0.1, 0.1) * h)
            scale = random.uniform(0.9, 1.1)
            shear = random.uniform(-10, 10)
            img_tensor, coords = self.apply_affine(img_tensor, coords, angle, translate, scale, shear)

            if random.random() < 0.5:
                img_tensor, coords = self.apply_perspective(img_tensor, coords, distortion_scale=0.2)

            img_tensor = self.color_aug(img_tensor)

        heatmap = self.coords_to_heatmap(coords, img_tensor.shape[1:], sigma=self.sigma)
        return img_tensor, heatmap

    def apply_random_crop(self, img_tensor, coords, crop_size=256):
        _, h, w = img_tensor.shape
        if h <= crop_size or w <= crop_size:
            return img_tensor, coords

        x0 = random.randint(0, w - crop_size)
        y0 = random.randint(0, h - crop_size)
        x1, y1 = x0 + crop_size, y0 + crop_size

        img_cropped = img_tensor[:, y0:y1, x0:x1]
        coords_cropped = coords - np.array([x0, y0])

        mask = (
                (coords_cropped[:, 0] >= 0) & (coords_cropped[:, 0] < crop_size) &
                (coords_cropped[:, 1] >= 0) & (coords_cropped[:, 1] < crop_size)
        )
        coords_cropped = coords_cropped[mask]

        return img_cropped, coords_cropped

    def apply_affine(self, img_tensor, coords, angle=0, translate=(0, 0), scale=1.0, shear=0.0):
        img_np = (img_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        h, w = img_np.shape
        center = (w / 2.0, h / 2.0)

        M = cv2.getRotationMatrix2D(center, angle, scale)
        shear_rad = np.deg2rad(shear)
        shear_mat = np.array([[1, -np.tan(shear_rad)], [0, 1]])
        M[:2, :2] = shear_mat @ M[:2, :2]
        M[:, 2] += translate

        img_warped = cv2.warpAffine(img_np, M, (w, h), borderValue=255)

        coords_hom = np.hstack([coords, np.ones((coords.shape[0], 1))])
        coords_warped = (M @ coords_hom.T).T

        img_tensor = torch.from_numpy(img_warped).unsqueeze(0).float() / 255.0

        coords_int = coords_warped.astype(int)
        valid_mask = []
        for x, y in coords_int:
            if 0 <= x < w and 0 <= y < h and img_tensor[0, y, x] < 0.99:
                valid_mask.append(True)
            else:
                valid_mask.append(False)

        coords_warped = coords_warped[valid_mask]
        return img_tensor, coords_warped

    def apply_perspective(self, img_tensor, coords, distortion_scale=0.2):
        img_np = (img_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        h, w = img_np.shape

        def random_shift(pt):
            dx = np.random.uniform(-distortion_scale, distortion_scale) * w
            dy = np.random.uniform(-distortion_scale, distortion_scale) * h
            return [pt[0] + dx, pt[1] + dy]

        src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        dst_pts = np.array([random_shift(pt) for pt in src_pts], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_np = cv2.warpPerspective(img_np, M, (w, h), borderValue=255)

        coords_hom = np.hstack([coords, np.ones((coords.shape[0], 1))])
        coords_proj = (M @ coords_hom.T).T
        coords_proj /= coords_proj[:, [2]]
        coords_warped = coords_proj[:, :2]

        img_tensor = torch.from_numpy(warped_np).unsqueeze(0).float() / 255.0

        coords_int = coords_warped.astype(int)
        valid_mask = []
        for x, y in coords_int:
            if 0 <= x < w and 0 <= y < h and img_tensor[0, y, x] < 0.99:
                valid_mask.append(True)
            else:
                valid_mask.append(False)

        coords_warped = coords_warped[valid_mask]
        return img_tensor, coords_warped

    def coords_to_heatmap(self, coords, img_size, sigma=2):
        H, W = img_size
        heatmap = torch.zeros((1, H, W), dtype=torch.float32)
        tmp_size = int(3 * sigma)

        for x, y in coords:
            x = int(round(x))
            y = int(round(y))
            if x < 0 or y < 0 or x >= W or y >= H:
                continue

            x0 = max(0, x - tmp_size)
            x1 = min(W, x + tmp_size + 1)
            y0 = max(0, y - tmp_size)
            y1 = min(H, y + tmp_size + 1)

            yy, xx = torch.meshgrid(
                torch.arange(y0, y1, dtype=torch.float32),
                torch.arange(x0, x1, dtype=torch.float32),
                indexing='ij'
            )
            g = torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
            g = g / g.max()

            heatmap[0, y0:y1, x0:x1] = torch.maximum(heatmap[0, y0:y1, x0:x1], g)

        return heatmap

    def apply_random_zoom(self, img_tensor, coords, zoom_range=(0.8, 1.2)):
        c, h, w = img_tensor.shape
        scale = random.uniform(*zoom_range)
        new_h = int(h * scale)
        new_w = int(w * scale)

        img_np = (img_tensor.squeeze(0).numpy() * 255).astype(np.uint8)
        img_resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        if scale > 1.0:
            x0 = (new_w - w) // 2
            y0 = (new_h - h) // 2
            img_zoomed = img_resized[y0:y0 + h, x0:x0 + w]
            offset = np.array([-x0, -y0])
        else:
            pad_left = (w - new_w) // 2
            pad_top = (h - new_h) // 2
            img_zoomed = np.full((h, w), 255, dtype=np.uint8)
            img_zoomed[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = img_resized
            offset = np.array([pad_left, pad_top])

        coords = coords * scale + offset

        img_tensor = torch.from_numpy(img_zoomed).unsqueeze(0).float() / 255.0

        coords_int = coords.astype(int)
        valid_mask = []
        for x, y in coords_int:
            if 0 <= x < w and 0 <= y < h and img_tensor[0, y, x] < 0.99:
                valid_mask.append(True)
            else:
                valid_mask.append(False)

        coords = coords[valid_mask]
        return img_tensor, coords
