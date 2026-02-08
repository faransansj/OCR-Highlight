#!/usr/bin/env python3
"""
YOLOv8 Ultra High Performance Training for NVIDIA RTX A6000
- Architecture: YOLOv8l (Large)
- Resolution: 1280px
- Device: NVIDIA GPU with 48GB VRAM
"""
from ultralytics import YOLO
import os

print("=" * 60)
print("⚔️  NVIDIA RTX A6000: Excalibur Training Initiation")
print("=" * 60)

# 1. Load Large Model (Higher intelligence, more parameters)
model = YOLO('yolov8l.pt') 

# 2. Start Full Power Training
# A6000's 48GB VRAM can easily handle imgsz=1280 and large batches
results = model.train(
    data='data/yolo_dataset/dataset.yaml',      # Ensure paths are correct
    epochs=100,                    # Sufficient iterations for deep learning
    imgsz=1280,                    # Ultra high resolution for small markups
    batch=16,                      # Reduced from 32 to avoid CUDA OOM with Large model at 1280px
    device=0,                      # Primary NVIDIA GPU
    project='markup_detector_a6000',
    name='v2_large_model_1280',
    patience=20,                   # Be more patient with a larger model
    save=True,
    amp=True,                      # Automatic Mixed Precision for speed
    mosaic=1.0,                    # Heavy data augmentation
    mixup=0.2,                     # Advanced augmentation
    workers=6,                     # Recommended number of workers for this system
    verbose=True
)

print("\n✅ Excalibur Training Completed!")
print(f"Legendary Model: markup_detector_a6000/v2_large_model_1280/weights/best.pt")
