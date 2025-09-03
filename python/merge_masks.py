from PIL import Image
import numpy as np
import os

input_mask_dir = "E:/Master_AI_project/Basilbot/data/masks_raw"
output_mask_dir = "E:/Master_AI_project/Basilbot/data/masks"

os.makedirs(output_mask_dir, exist_ok=True)

for fname in os.listdir(input_mask_dir):
    if not fname.endswith(".png"):
        continue

    base_name = fname.rsplit("_", 1)[0]
    mask_path = os.path.join(input_mask_dir, fname)
    mask = np.array(Image.open(mask_path).convert("L")) > 0

    merged_path = os.path.join(output_mask_dir, base_name + ".png")
    if os.path.exists(merged_path):
        merged = np.array(Image.open(merged_path).convert("L")) > 0
        mask = np.logical_or(mask, merged)

    Image.fromarray((mask * 255).astype(np.uint8)).save(merged_path)
