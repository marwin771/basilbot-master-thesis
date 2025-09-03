import os
import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image

# === Paths ===
image_dir = "E:/Master_AI_project/Basilbot/data/images"
mask_dir = "E:/Master_AI_project/Basilbot/data/masks_raw"
checkpoint_path = "E:/Master_AI_project/Basilbot/sam_vit_h_4b8939.pth"
model_type = "vit_h"

# === Load SAM ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam.to(device)
sam.eval()
predictor = SamPredictor(sam)

# === Helper ===
def get_next_mask_filename(base_name):
    counter = 1
    while True:
        fname = f"{base_name}_{counter:03d}.png"
        full_path = os.path.join(mask_dir, fname)
        if not os.path.exists(full_path):
            return full_path
        counter += 1

# === GUI Loop ===
image_files = [f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
image_index = 0
click_coords = []

while image_index < len(image_files):
    filename = image_files[image_index]
    img_path = os.path.join(image_dir, filename)
    image_bgr = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    predictor.set_image(image_rgb)
    click_coords = []

    scale_width = 800
    scale_factor = scale_width / w
    display_size = (scale_width, int(h * scale_factor))

    def mouse_callback(event, x, y, flags, param):
        global click_coords
        if event == cv2.EVENT_LBUTTONDOWN:
            full_x = int(x / scale_factor)
            full_y = int(y / scale_factor)
            click_coords = [[full_x, full_y]]

    cv2.namedWindow("Click on Petiole")
    cv2.setMouseCallback("Click on Petiole", mouse_callback)

    mask_display = image_bgr.copy()
    while True:
        vis = mask_display.copy()
        if click_coords:
            input_point = np.array(click_coords)
            input_label = np.array([1])
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False
            )
            mask = masks[0].astype(np.uint8) * 255
            overlay = image_bgr.copy()
            overlay[mask > 0] = (0, 0, 255)
            vis = cv2.addWeighted(image_bgr, 0.7, overlay, 0.3, 0)

        vis_resized = cv2.resize(vis, display_size)
        cv2.putText(vis_resized, f"Image {image_index+1}/{len(image_files)} - {filename}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_resized, "[S] Save  [N] Next  [Q] Quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.imshow("Click on Petiole", vis_resized)
        key = cv2.waitKey(10)

        if key == ord('s') and click_coords:
            base_name = os.path.splitext(filename)[0]
            mask_filename = get_next_mask_filename(base_name)
            Image.fromarray(mask).save(mask_filename)
            print(f"💾 Saved mask: {mask_filename}")

        elif key == ord('n'):
            image_index += 1
            break

        elif key == ord('q'):
            image_index = len(image_files)
            break

cv2.destroyAllWindows()
