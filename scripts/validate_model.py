#!/usr/bin/env python3
"""
Validate the best model checkpoint
"""
from ultralytics import YOLO

print("=" * 60)
print("📊 Model Validation")
print("=" * 60)

# Load best model
model = YOLO('runs/detect/runs/detect/markup_detector5/weights/best.pt')

# Validate on test set
results = model.val(
    data='data/yolo_dataset_preprocessed/dataset.yaml',
    split='val',
    batch=8,
    device='cpu',
    verbose=True
)

print("\n" + "=" * 60)
print("✅ Validation Results:")
print("=" * 60)
print(f"mAP50: {results.box.map50:.4f} ({results.box.map50*100:.2f}%)")
print(f"mAP50-95: {results.box.map:.4f} ({results.box.map*100:.2f}%)")
print(f"Precision: {results.box.p[0]:.4f}")
print(f"Recall: {results.box.r[0]:.4f}")
print("=" * 60)
