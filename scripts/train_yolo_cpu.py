#!/usr/bin/env python3
"""
YOLOv8 CPU Training - Final Stable Version
- No GPU dependencies
- Optimized for 30-epoch completion
- Memory-efficient batch processing
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU

from ultralytics import YOLO

print("=" * 60)
print("🖥️  CPU-Only YOLO Training (Stable)")
print("=" * 60)

# Load model
model = YOLO('yolov8n.pt')

# Train on CPU
results = model.train(
    data='data/yolo_dataset_preprocessed/dataset.yaml',
    epochs=30,        # Reduced for stability
    imgsz=640,
    batch=4,          # Conservative for CPU
    device='cpu',
    project='markup_detector_cpu',
    name='final_run',
    patience=10,
    save=True,
    save_period=5,    # Checkpoint every 5 epochs
    cache=False,
    workers=4,
    verbose=True
)

print("\n✅ Training completed!")
print(f"Best: markup_detector_cpu/final_run/weights/best.pt")
