import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
from segment_anything import sam_model_registry
from loralib import Linear as LoRALinear
import matplotlib.pyplot as plt
import pandas as pd

def apply_lora(module, r=4, lora_alpha=1.0, lora_dropout=0.0):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            in_features = child.in_features
            out_features = child.out_features
            bias = child.bias is not None
            new_layer = LoRALinear(in_features, out_features, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias=bias)
            new_layer.weight.data = child.weight.data.clone()
            if bias:
                new_layer.bias.data = child.bias.data.clone()
            setattr(module, name, new_layer)
        else:
            apply_lora(child, r, lora_alpha, lora_dropout)

# === Config ===
image_dir = "E:/Master_AI_project/Basilbot/data/images"
mask_dir = "E:/Master_AI_project/Basilbot/data/masks"
checkpoint_path = "E:/Master_AI_project/Basilbot/sam_vit_h_4b8939.pth"
model_type = "vit_h"
batch_size = 2
image_size = 1024
lr = 1e-4
epochs = 100
patience = 5
log_file = "training_log.txt"

# === Dataset ===
class PetioleDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.filenames = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
        self.image_size = image_size

        self.img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_path = os.path.join(self.image_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname.replace(".jpg", "").replace(".jpeg", "").replace(".png", "") + "_001.png")

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        return self.img_transform(img), self.mask_transform(mask)

# === Load Data ===
dataset = PetioleDataset(image_dir, mask_dir, image_size)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)

# === Load SAM + LoRA ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
apply_lora(sam.mask_decoder)
sam.to(device)
sam.train()

# === Optimizer ===
params = [p for p in sam.mask_decoder.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=lr)

# === Loss + Metrics ===
def compute_loss(pred_mask, true_mask):
    pred_mask = torch.sigmoid(pred_mask)
    return F.binary_cross_entropy(pred_mask, true_mask)

def dice_coefficient(pred, target, epsilon=1e-6):
    pred = (torch.sigmoid(pred) > 0.5).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum(dim=[1,2,3])
    union = pred.sum(dim=[1,2,3]) + target.sum(dim=[1,2,3])
    dice = (2 * intersection + epsilon) / (union + epsilon)
    return dice.mean().item()

# === Early stopping initialization ===
best_val_loss = float('inf')
best_train_loss = float('inf')
best_state = None
patience_counter = 0

# === Training Loop ===
train_losses, val_losses = [] , []
train_dices, val_dices = [], []

with open(log_file, 'w') as log:
    for epoch in range(epochs):
        sam.train()
        train_loss, train_dice = 0, 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.no_grad():
                img_embeddings = sam.image_encoder(imgs)
                sparse_embeddings, dense_embeddings = sam.prompt_encoder(points=None, boxes=None, masks=None)
            low_res_masks = sam.mask_decoder(
                image_embeddings=img_embeddings,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )[0]
            upsampled = F.interpolate(low_res_masks, size=(image_size, image_size), mode="bilinear", align_corners=False)
            loss = compute_loss(upsampled, masks)
            dice = dice_coefficient(upsampled, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_dice += dice

        train_loss /= len(train_loader)
        train_dice /= len(train_loader)

        sam.eval()
        val_loss, val_dice = 0, 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                img_embeddings = sam.image_encoder(imgs)
                sparse_embeddings, dense_embeddings = sam.prompt_encoder(points=None, boxes=None, masks=None)
                low_res_masks = sam.mask_decoder(
                    image_embeddings=img_embeddings,
                    image_pe=sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False
                )[0]
                upsampled = F.interpolate(low_res_masks, size=(image_size, image_size), mode="bilinear", align_corners=False)
                loss = compute_loss(upsampled, masks)
                dice = dice_coefficient(upsampled, masks)
                val_loss += loss.item()
                val_dice += dice

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_dices.append(train_dice)
        val_dices.append(val_dice)

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")
        log.write(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Dice={train_dice:.4f} | Val Loss={val_loss:.4f}, Dice={val_dice:.4f}\n")

        if val_loss < best_val_loss or train_loss < best_train_loss:
            best_val_loss = min(val_loss, best_val_loss)
            best_train_loss = min(train_loss, best_train_loss)
            best_state = sam.mask_decoder.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

# === Save best model ===
torch.save(best_state, "lora_sam_mask_decoder.pt")
print("✅ LoRA weights saved as lora_sam_mask_decoder.pt")

# === Export CSV ===
log_df = pd.DataFrame({
    "epoch": list(range(1, len(train_losses) + 1)),
    "train_loss": train_losses,
    "val_loss": val_losses,
    "train_dice": train_dices,
    "val_dice": val_dices
})
log_df.to_csv("training_metrics.csv", index=False)
print("🧾 Saved metrics to training_metrics.csv")

# === Plotting ===
epochs_ran = len(train_losses)
plt.figure()
plt.plot(range(1, epochs_ran + 1), train_losses, label='Train Loss')
plt.plot(range(1, epochs_ran + 1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig("loss_curve.png")

plt.figure()
plt.plot(range(1, epochs_ran + 1), train_dices, label='Train Dice')
plt.plot(range(1, epochs_ran + 1), val_dices, label='Val Dice')
plt.xlabel('Epoch')
plt.ylabel('Dice Coefficient')
plt.title('Dice Score Curve')
plt.legend()
plt.grid(True)
plt.savefig("dice_curve.png")

print("📉 Saved loss curve as loss_curve.png")
print("📈 Saved dice curve as dice_curve.png")