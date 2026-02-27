# 📔 Project Development Log

Historical summary and status updates for the OCR Highlight project.

---

## 📍 Current Status (2026-02-07)

### ✅ Completed
- Dataset generation (9k training images).
- Image preprocessing (640px optimization).
- Base model training (Epoch 1 checkpoint available).
- Universal inference script (`predict.py`).
- GitHub authentication via Deploy Key.

### ⚠️ Blocked
- AMD GPU (Radeon 780M) native training due to ROCm/PyTorch ISA incompatibility with gfx1103.

### 🎯 Next Goal
- Full 50-epoch training on **M4 Apple Silicon** using `mps` backend.

---

## 🕰️ Timeline Highlights

- **Phase 1**: Dataset generation (Week 1-2). 600 synthetic images created.
- **Phase 2**: Highlight detection logic optimized (Week 3-4). mIoU 0.82.
- **Phase 3**: OCR integration and post-processing (Week 5-6). 95% accuracy reached.
- **Phase 4**: Transition to YOLO Deep Learning (Present). Reached 96% mAP50.

---

[⬅ Back to README](../README.md)
