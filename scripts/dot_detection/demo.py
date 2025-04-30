import os

import matplotlib.cm as cm
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from dataloader import BrailleDataset

# Output directory for debug images
output_dir = "./debug_output"
os.makedirs(output_dir, exist_ok=True)

# Create dataset and dataloader
dataset = BrailleDataset(root_dir="./prepared-patches", transform_prob=0.2)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Iterate and save debug patches
for idx, (img_tensor, heatmap_tensor) in enumerate(loader):
    img = F.to_pil_image(img_tensor[0])  # Convert image tensor to PIL

    # Convert heatmap to grayscale image
    heatmap = heatmap_tensor[0, 0]  # Shape: H x W
    heatmap_np = heatmap.clamp(0, 1).numpy()
    heatmap_img = Image.fromarray((cm.inferno(heatmap_np)[:, :, :3] * 255).astype('uint8'))  # Apply colormap

    # Resize heatmap to match image size (safety)
    heatmap_img = heatmap_img.resize(img.size, resample=Image.BILINEAR)

    # Option A: Combine side by side
    combined = Image.new("RGB", (img.width * 2, img.height))
    combined.paste(img, (0, 0))
    combined.paste(heatmap_img, (img.width, 0))

    # Option B: Or overlay heatmap on image (if you prefer)
    # overlay = Image.blend(img.convert("RGB"), heatmap_img, alpha=0.5)

    combined.save(os.path.join(output_dir, f"debug_img_{idx:05d}.png"))

    if idx >= 49:
        break  # Save first 50 only

print(f"Saved {idx + 1} debug images to {output_dir}")
