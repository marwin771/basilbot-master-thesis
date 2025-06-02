import cv2
import torch
import numpy as np
import torch.nn.functional as F
from segment_anything import sam_model_registry, SamPredictor
from loralib import Linear as LoRALinear

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

# === Settings ===
stream_url = "http://192.168.1.214:81/stream"  # Change if needed
checkpoint_path = "E:/Master_AI_project/Basilbot/sam_vit_h_4b8939.pth"
lora_weights_path = "E:/Master_AI_project/Basilbot/lora_sam_mask_decoder.pt"
model_type = "vit_h"
image_size = 1024

# === Load Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
apply_lora(sam.mask_decoder)
sam.to(device)
sam.eval()
sam.mask_decoder.load_state_dict(torch.load(lora_weights_path, map_location=device))
sam.mask_decoder.to(device)
predictor = SamPredictor(sam)

# === Stream Setup ===
def open_stream():
    print("🔄 Connecting to stream...")
    cap = cv2.VideoCapture(stream_url)
    if cap.isOpened():
        print("✅ Stream connected.")
    return cap

cap = open_stream()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame. Reconnecting...")
            cap.release()
            cap = open_stream()
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (image_size, image_size))

        predictor.set_image(resized)
        h, w = image_size, image_size

        # Dummy center point — you can update this with a smart policy
        center_point = np.array([[w // 2, h // 2]])
        input_label = np.array([1])

        masks, scores, _ = predictor.predict(
            point_coords=center_point,
            point_labels=input_label,
            multimask_output=False
        )

        mask = masks[0].astype(np.uint8) * 255
        mask_bgr = np.zeros_like(frame)
        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        mask_bgr[:, :, 2] = mask_resized

        overlay = cv2.addWeighted(frame, 0.7, mask_bgr, 0.3, 0)
        cv2.imshow("Real-time Petiole Segmentation", overlay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("🛑 Interrupted.")

finally:
    cap.release()
    cv2.destroyAllWindows()
