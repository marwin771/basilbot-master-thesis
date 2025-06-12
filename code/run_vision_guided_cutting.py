import cv2
import torch
import numpy as np
import serial
import time
from PIL import Image
from segment_anything import sam_model_registry
import loralib as lora
from loralib import Linear as LoRALinear
import torch.nn.functional as F

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
        else:
            apply_lora(child, r, lora_alpha, lora_dropout)

# === Settings ===
#stream_url = "http://192.168.226.188:81/stream"
stream_url = "http://192.168.1.215:81/stream"
checkpoint_path = "E:/Master_AI_project/Basilbot/sam_vit_h_4b8939.pth"
lora_weights = "results/lora_sam_mask_decoder_26.pt"
model_type = "vit_h"
image_size = 1024
center_tolerance = 0.03
serial_port = "COM6"
skip_every = 30

# Area thresholds
close_enough_area = 30000
lost_target_area = 1000
alpha = 0.3  # EMA smoothing factor

# === Serial & SAM ===
ser = serial.Serial(serial_port, 115200, timeout=1)
time.sleep(2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam.eval().to(device)
apply_lora(sam.mask_decoder)
sam.mask_decoder.load_state_dict(torch.load(lora_weights, map_location=device))
sam.mask_decoder.to(device)

# === Stream ===
def open_stream():
    print("🔄 Connecting to stream...")
    cap = cv2.VideoCapture(stream_url)
    time.sleep(1)
    if cap.isOpened():
        print("✅ Stream connected.")
    return cap

cap = open_stream()

# === States ===
frame_count = 0
cut_x, cut_y = None, None
stopped = False
approaching = False
ema_area = 0
last_mask = None
done = False  # Whether final position has been reached

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame grab failed. Reconnecting...")
            cap.release()
            cap = open_stream()
            continue

        # Resize and prepare image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(rgb, (image_size, image_size))
        h, w = image_size, image_size

        #if not done and frame_count % skip_every == 0:
        if not done and frame_count > 30 and frame_count % skip_every == 0:
            print("🧠 Running SAM...")
            image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            image_tensor = image_tensor.to(device)

            with torch.no_grad():
                image_embeddings = sam.image_encoder(image_tensor)
                sparse_embeddings, dense_embeddings = sam.prompt_encoder(points=None, boxes=None, masks=None)
                low_res_masks = sam.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False
                )[0]

                upsampled = F.interpolate(low_res_masks, size=(image_size, image_size), mode="bilinear", align_corners=False)
                pred_mask = torch.sigmoid(upsampled).squeeze().cpu().numpy() > 0.5

            binary_mask = (pred_mask * 255).astype(np.uint8)
            area = np.sum(pred_mask)
            ema_area = alpha * area + (1 - alpha) * ema_area
            print(f"📐 Area: raw={area:.0f}, EMA={ema_area:.0f}")

            # Visualize mask
            mask_bgr = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
            mask_colored = np.zeros_like(mask_bgr)
            mask_colored[:, :, 2] = binary_mask
            mask_colored = cv2.resize(mask_colored, (frame.shape[1], frame.shape[0]))
            last_mask = mask_colored

            # Find center of largest component
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
            if num_labels > 1:
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                cut_x, cut_y = centroids[largest_label]
                norm_x = round(cut_x / w, 3)
                norm_y = round(cut_y / h, 3)
                dx = norm_x - 0.5
                dy = norm_y - 0.5
                print(f"📏 Offset dx={dx:.3f}, dy={dy:.3f}")

                if not stopped:
                    if abs(dx) > center_tolerance or abs(dy) > center_tolerance:
                        msg = f"{norm_x},{norm_y},0\n"
                        ser.write(msg.encode())
                        print(f"📤 Sent to ESP32: {msg.strip()}")
                    else:
                        print("✅ Centered. Begin approach.")
                        stopped = True
                        approaching = True

                elif stopped and approaching:
                    if ema_area < lost_target_area:
                        print("⚠️ Petiole lost. Halting.")
                        approaching = False
                    elif ema_area > close_enough_area:
                        print("✂️ Close enough. Sending CUT.")
                        ser.write(b"CUT\n")  # Safe to keep in case it's handled later
                        stopped = False
                        approaching = False
                        done = True  # Freeze tracking
                    else:
                        print("➡️ Not close — sending FORWARD_STEP.")
                        ser.write(b"FORWARD_STEP\n")
                        time.sleep(0.1)
                        while ser.in_waiting:
                            line = ser.readline().decode().strip()
                            if line:
                                print(f"🔧 ESP32: {line}")
                        time.sleep(0.4)

        # === Draw overlay ===
        overlay = frame.copy()
        if last_mask is not None:
            overlay = cv2.addWeighted(frame, 0.7, last_mask, 0.3, 0)

        # Draw center marker and target point
        img_h, img_w = frame.shape[:2]
        cv2.drawMarker(overlay, (img_w // 2, img_h // 2), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

        if cut_x is not None and cut_y is not None:
            draw_x = int(cut_x * frame.shape[1] / w)
            draw_y = int(cut_y * frame.shape[0] / h)
            cv2.circle(overlay, (draw_x, draw_y), 8, (0, 0, 255), -1)

        # If done, freeze and show message
        if done:
            cv2.putText(overlay, "✅ Ready to cut!", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        cv2.imshow("Petiole Tracker", overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

except KeyboardInterrupt:
    print("🛑 Interrupted by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    ser.close()
