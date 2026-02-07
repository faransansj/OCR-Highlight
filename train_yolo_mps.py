#!/usr/bin/env python3
"""
YOLOv8 Training on Apple Silicon (M4) with MPS
"""
import torch
from ultralytics import YOLO

print("=" * 60)
print("🍎 Apple Silicon (M4) YOLO Training")
print("=" * 60)
print(f"PyTorch: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
print("=" * 60)

# Load pretrained model
model = YOLO('yolov8n.pt')

# Train with MPS (Apple Silicon GPU)
results = model.train(
    data='data/yolo_dataset_preprocessed/dataset.yaml',
    epochs=50,
    imgsz=640,
    batch=16,          # Adjust based on available memory
    device='mps',      # Use Apple Metal Performance Shaders
    project='markup_detector_mps',
    name='m4_training',
    patience=10,
    save=True,
    save_period=5,     # Save checkpoint every 5 epochs
    cache=False,
    workers=4,
    amp=False,         # AMP not needed on MPS
    verbose=True
)

print("\n🎉 Training completed!")
print(f"Best model: markup_detector_mps/m4_training/weights/best.pt")
