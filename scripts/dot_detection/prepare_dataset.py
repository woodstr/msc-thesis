import csv
import os

import cv2
import numpy as np


def image_rotate(image, angle_deg):
    """Rotate the image counterclockwise by `angle_deg` degrees."""
    (h, w) = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    scale = 1.0
    angle_rad = np.deg2rad(angle_deg)
    a = np.sin(angle_rad) * scale
    b = np.cos(angle_rad) * scale
    new_w = int(h * abs(a) + w * abs(b))
    new_h = int(w * abs(a) + h * abs(b))

    M = cv2.getRotationMatrix2D(center, -angle_deg, scale)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    return rotated, M


def split_and_save_patches(image, dots, output_dir, base_index, patch_size=256):
    """Split image and dot coordinates into NxN patches and save each as img_xxxxx.*"""
    os.makedirs(output_dir, exist_ok=True)
    h, w = image.shape[:2]
    patch_idx = 0

    for y0 in range(0, h, patch_size):
        for x0 in range(0, w, patch_size):
            x1 = min(x0 + patch_size, w)
            y1 = min(y0 + patch_size, h)
            patch = image[y0:y1, x0:x1]

            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                continue  # Skip incomplete patch

            local_dots = []
            for x, y in dots:
                if x0 <= x < x1 and y0 <= y < y1:
                    local_dots.append((x - x0, y - y0))

            name = f"img_{base_index + patch_idx:05d}"
            img_path = os.path.join(output_dir, name + ".png")
            csv_path = os.path.join(output_dir, name + ".csv")

            patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(img_path, patch_gray)

            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["x", "y"])
                writer.writerows(local_dots)

            patch_idx += 1

    return patch_idx


def parse_and_save(image_path, output_dir, base_index, patch_size=256):
    """Parse annotation and split into patches with adjusted coordinates."""
    base_path = os.path.splitext(image_path)[0]
    recto_path = base_path + "+recto.txt"
    verso_path = base_path + "+verso.txt"

    image = cv2.imread(image_path)
    all_coords = []
    rotated_image = image
    last_transform = None

    for txt_path in [recto_path, verso_path]:
        if not os.path.exists(txt_path):
            continue

        with open(txt_path, 'r') as f:
            lines = f.readlines()

        if len(lines) < 3:
            continue

        angle = float(lines[0].strip())
        verticals = list(map(int, lines[1].strip().split()))
        horizontals = list(map(int, lines[2].strip().split()))
        cell_lines = lines[3:]

        rotated_image, transform = image_rotate(image, -angle)
        last_transform = transform

        for cell in cell_lines:
            parts = list(map(int, cell.strip().split()))
            r, c = parts[0] - 1, parts[1] - 1
            dots = parts[2:]
            for i, val in enumerate(dots):
                if val == 1:
                    if i < 3:
                        y = horizontals[r * 3 + i]
                        x = verticals[c * 2]
                    else:
                        y = horizontals[r * 3 + i - 3]
                        x = verticals[c * 2 + 1]
                    coord = np.array([[x, y]], dtype=np.float32)
                    coord = np.array([coord])
                    rotated_coord = cv2.transform(coord, transform)[0][0]
                    all_coords.append(rotated_coord.tolist())

        image = rotated_image

    if last_transform is None:
        return 0

    return split_and_save_patches(image, all_coords, output_dir, base_index, patch_size)


def collect_images(root_dir):
    """Recursively collect image files excluding those containing '+recto' or '+verso'."""
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    file_list = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(valid_extensions):
                if "+recto" not in fname and "+verso" not in fname:
                    full_path = os.path.join(dirpath, fname)
                    file_list.append(full_path)
    return file_list


if __name__ == "__main__":
    root_directory = "./DSBI"
    output_directory = "./prepared-patches"
    patch_size = 512

    image_list = collect_images(root_directory)
    print(f"Found {len(image_list)} images to process.")

    os.makedirs(output_directory, exist_ok=True)

    patch_counter = 0
    for path in sorted(image_list):
        print(f"Processing {path}...")
        n = parse_and_save(path, output_directory, base_index=patch_counter, patch_size=patch_size)
        patch_counter += n
