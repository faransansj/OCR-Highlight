#!/usr/bin/env python3
"""
Markup Detection Inference Script
Detects highlights, underlines, strikethrough, circles, and rectangles in document images
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU inference

import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2

# Class names mapping
CLASS_NAMES = {
    0: 'highlight',
    1: 'underline', 
    2: 'strikethrough',
    3: 'circle',
    4: 'rectangle'
}

def predict_image(model, image_path, conf_threshold=0.25, save_dir='predictions'):
    """
    Run inference on a single image
    
    Args:
        model: YOLO model
        image_path: Path to input image
        conf_threshold: Confidence threshold (0-1)
        save_dir: Directory to save results
    
    Returns:
        dict: Detection results with counts per class
    """
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Run prediction
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        save=True,
        project=save_dir,
        name='',
        exist_ok=True
    )
    
    # Parse results
    detections = {name: 0 for name in CLASS_NAMES.values()}
    boxes = results[0].boxes
    
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = CLASS_NAMES.get(cls_id, 'unknown')
            detections[class_name] += 1
            
            print(f"  {class_name}: {conf:.3f}")
    
    return detections, results[0]

def main():
    parser = argparse.ArgumentParser(description='Detect markup in document images')
    parser.add_argument('image', type=str, help='Path to input image')
    parser.add_argument('--model', type=str, default='final_model/markup_detector_v2_m4.pt',
                        help='Path to model weights')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    parser.add_argument('--save-dir', type=str, default='predictions',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    
    # Run inference
    print(f"\nProcessing: {args.image}")
    detections, result = predict_image(model, args.image, args.conf, args.save_dir)
    
    # Summary
    print("\n" + "=" * 60)
    print("Detection Summary:")
    print("=" * 60)
    total = sum(detections.values())
    for class_name, count in detections.items():
        if count > 0:
            print(f"  {class_name}: {count}")
    print(f"\nTotal detections: {total}")
    print(f"Results saved to: {args.save_dir}/")
    print("=" * 60)

if __name__ == '__main__':
    main()
