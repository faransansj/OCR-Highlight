#!/usr/bin/env python3
"""
Validate the Large YOLOv8 model
"""
import os
import argparse
from ultralytics import YOLO

def main(model_path, data_config):
    print("=" * 60)
    print(f"📊 Model Validation: {os.path.basename(model_path)}")
    print("=" * 60)

    # Load model
    model = YOLO(model_path)

    # Validate
    results = model.val(
        data=data_config,
        split='val',
        batch=4, # Large model uses more VRAM/Memory
        device='cpu', # Default to CPU for stability in this env
        verbose=True
    )

    print("\n" + "=" * 60)
    print("✅ Validation Results Summary:")
    print("=" * 60)
    print(f"mAP50:    {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    
    # Class-wise stats
    names = model.names
    for i, name in names.items():
        if i < len(results.box.maps):
            print(f"[{name}] mAP50: {results.box.maps[i]:.4f}")

    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="final_model/large_v1/markup_detector_l_v1.pt")
    parser.add_argument("--data", type=str, default="data/yolo_dataset_preprocessed/dataset.yaml")
    args = parser.parse_args()
    
    main(args.model, args.data)
