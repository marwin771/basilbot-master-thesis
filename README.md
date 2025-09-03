# Vision-Guided Basil Harvester (BasilBot)

A low-cost, RGB-camera-based robotic system for **precision basil leaf harvesting**, powered by a fine-tuned SAM model and controlled via an ESP32-based robotic arm. The system detects petioles, centers the cutter, approaches, and performs precise cuts — all in real time.

---

## 🎯 Features

* 🧠 Real-time petiole segmentation using LoRA-fine-tuned [SAM (Segment Anything)](https://github.com/facebookresearch/segment-anything)
* 📵 Live ESP32-CAM video stream
* 🤖 ESP32-controlled 5DOF robotic arm with smooth servo motion
* ✂️ Autonomous cutting based on image mask area + spatial alignment
* 📉 EMA-based smoothing for mask area to handle occlusion

---

## 🛠️ System Overview

| Component          | Description                                        |
| ------------------ | -------------------------------------------------- |
| **ESP32-S3**       | Controls servo motors + camera stream              |
| **OV2640 Camera**  | Mounted on wrist for vision-based control          |
| **LoRA-SAM Model** | Detects petiole segmentation mask                  |
| **Python Tracker** | Processes stream, computes target & sends commands |
| **Servo Arm**      | 5DOF arm with base, shoulder, elbow, wrist, cutter |

---

## 📁 Project Structure

```
vision-guided-basil-harvester/
├── arduino/                 # ESP32 firmware sketch (.ino)
├── python/                  # Vision + control logic
├── models/                  # SAM + LoRA weights
├── dataset/                 # Training images/masks
├── recordings/              # Video outputs from robot runs
├── docs/                    # Diagrams and visualizations
└── README.md
```

---

## 🤖 Control Logic

### 📌 Centering Phase

* Tracks `(x, y)` offset of petiole mask center
* Sends coordinates to ESP32 to align base and shoulder

### 🔀 Approaching Phase

* Once centered, sends `FORWARD_STEP` repeatedly
* Moves **shoulder**, **elbow**, and **wrist** forward
* Uses **EMA-smoothed mask area** to decide proximity

### ✂️ Cutting Phase

* If mask is **close enough**, sends `CUT`
* Cutter motor moves to `0` and returns to resting position

---

## 🧠 Technical Highlights
* **LoRA Fine-tuning of SAM**: The Segment Anything Model (SAM) was fine-tuned with LoRA on custom petiole segmentation masks to enable precise detection in basil environments.
* **EMA Filtering**: Smooths out noisy mask area changes
* **Occlusion Tolerance**: If petiole is lost after centering, still proceeds with cut
* **Recorded Videos**: Save raw and annotated views for analysis

