import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import BrailleDataset
from model import UNet

# === Config ===
batch_size = 8
num_epochs = 30
lr = 1e-3
train_val_ratio = 0.8
save_dir = "./checkpoints"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(save_dir, exist_ok=True)

# === Load dataset ===
train_dataset = BrailleDataset(root_dir="./prepared-patches", is_train=True, train_val_ratio=train_val_ratio)
val_dataset = BrailleDataset(root_dir="./prepared-patches", is_train=False, train_val_ratio=train_val_ratio)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# === Model ===
model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.L1Loss()

best_val_loss = float('inf')
for epoch in range(1, num_epochs + 1):
    model.train()
    train_loss = 0

    for img, heatmap in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]"):
        img = img.to(device)
        heatmap = heatmap.to(device)

        pred = model(img)
        loss = criterion(pred, heatmap)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * img.size(0)

    train_loss /= len(train_loader.dataset)

    # === Validation ===
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for img, heatmap in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]"):
            img = img.to(device)
            heatmap = heatmap.to(device)

            pred = model(img)
            loss = criterion(pred, heatmap)

            val_loss += loss.item() * img.size(0)

    val_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        ckpt_path = os.path.join(save_dir, f"unet_best.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved best model to {ckpt_path}")
