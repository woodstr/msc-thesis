import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_tensor

from model import UNet

# === Config ===
image_dir = "./test-data"
output_dir = "./results"
model_path = "./checkpoints/unet_best.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sigma = 3
peak_threshold = 0.5
input_size = 384  # NxN target input

os.makedirs(output_dir, exist_ok=True)

# === Load model ===
model = UNet(in_channels=1, out_channels=1).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


# === Retinex reflectance extractor ===
def compute_retinex_reflectance(img_pil, sigma=30):
    img_np = np.array(img_pil).astype(np.float32) + 1.0  # Avoid log(0)
    log_img = np.log(img_np)

    blurred = cv2.GaussianBlur(img_np, (0, 0), sigmaX=sigma, sigmaY=sigma)
    log_blur = np.log(blurred + 1.0)

    reflectance = log_img - log_blur
    reflectance = (reflectance - reflectance.min()) / (reflectance.max() - reflectance.min()) * 255.0
    reflectance = reflectance.astype(np.uint8)
    return Image.fromarray(reflectance)


# === NMS ===
def extract_peaks_from_heatmap(heatmap, threshold=0.5, dist=3):
    heatmap = heatmap.squeeze(0)
    pooled = F.max_pool2d(heatmap.unsqueeze(0).unsqueeze(0), kernel_size=2 * dist + 1, stride=1, padding=dist)
    peak_mask = (heatmap == pooled.squeeze()) & (heatmap > threshold)
    coords = peak_mask.nonzero(as_tuple=False)  # (y, x)
    coords = coords[:, [1, 0]].cpu().numpy()  # (x, y)
    return coords


# === Load all test images ===
image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
img_paths = []
for ext in image_extensions:
    img_paths.extend(glob.glob(os.path.join(image_dir, ext)))
img_paths = sorted(img_paths)

# === Run inference ===
for idx, img_path in enumerate(img_paths):
    original_img = Image.open(img_path).convert("L")
    original_img_resized = original_img.resize((input_size, input_size), Image.BILINEAR)

    # Compute reflectance map via Retinex
    reflectance_img = compute_retinex_reflectance(original_img_resized, sigma=50)
    img_tensor = to_tensor(reflectance_img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_tensor)  # [1, 1, N, N]
        pred = pred.squeeze(0).cpu()

    # Extract dot coords in resized image
    coords = extract_peaks_from_heatmap(pred, threshold=peak_threshold, dist=int(sigma * 1.5))

    # Draw on reflectance image
    reflectance_with_boxes = reflectance_img.convert("RGB")
    draw = ImageDraw.Draw(reflectance_with_boxes)
    for x, y in coords:
        draw.rectangle([x - 3, y - 3, x + 3, y + 3], outline="green", width=2)

    # === Save side-by-side plot with 3 panels ===
    base_name = os.path.basename(img_path)
    heatmap_save_path = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}_viz.png")

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(original_img_resized, cmap="gray")
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(reflectance_with_boxes)
    axs[1].set_title("Reflectance + Prediction")
    axs[1].axis("off")

    axs[2].imshow(pred.squeeze(), cmap="hot", interpolation="nearest")
    axs[2].set_title("Predicted Heatmap")
    axs[2].axis("off")

    plt.tight_layout()
    plt.savefig(heatmap_save_path)
    plt.close(fig)

print(f"Saved {len(img_paths)} prediction results with visualizations to {output_dir}")
