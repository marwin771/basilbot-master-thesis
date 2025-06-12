import os
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from loralib import Linear as LoRALinear

# === LoRA loader ===
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
image_path = "E:/Master_AI_project/Basilbot/data/20250610190618.jpg"  # <- Change if needed
checkpoint_path = "E:/Master_AI_project/Basilbot/sam_vit_h_4b8939.pth"
lora_weights = "results/lora_sam_mask_decoder_26.pt"
model_type = "vit_h"
image_size = 1024
threshold = 0.5  # Lower if masks are too faint
save_path = "test_lora_sam_result.png"

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

# === Load SAM and apply LoRA ===
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
apply_lora(sam.mask_decoder)
sam.mask_decoder.load_state_dict(torch.load(lora_weights, map_location=device))
sam.to(device).eval()
predictor = SamPredictor(sam)

# === Load and prepare image ===
image = Image.open(image_path).convert("RGB")
image = image.resize((image_size, image_size))
image_np = np.array(image)
predictor.set_image(image_np)

# === Use central point as prompt ===
h, w, _ = image_np.shape
input_point = np.array([[w // 2, h // 2]])
input_label = np.array([1])

# === Predict mask ===
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=False
)

# === Post-process mask ===
soft_mask = logits[0]  # raw logits
sigmoid_mask = torch.sigmoid(torch.from_numpy(soft_mask)).numpy()
binary_mask = (sigmoid_mask > threshold).astype(np.uint8) * 255
binary_mask = cv2.resize(binary_mask, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)

# === Overlay visualization ===
overlay = image_np.copy()
overlay[binary_mask > 0] = (255, 0, 0)  # Red overlay for mask
blended = cv2.addWeighted(image_np, 0.7, overlay, 0.3, 0)

# === Visualization ===
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.imshow(image_np)
plt.title("Input Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(sigmoid_mask, cmap='hot')
plt.title("Soft Mask (Sigmoid)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(blended)
plt.title(f"Binary Mask Overlay (thr={threshold})")
plt.axis("off")

plt.tight_layout()
plt.savefig(save_path)
plt.show()
print(f"✅ Prediction complete — saved as {save_path}")
