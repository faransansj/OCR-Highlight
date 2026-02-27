#!/usr/bin/env python3
"""
Test Highlight Detection on Real World Images
Basic baseline test for Milestone 2, Phase 2.1
"""

import sys
import os
import cv2
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from highlight_detector import HighlightDetector

def test_real_images():
    print("\n" + "=" * 60)
    print("REAL WORLD HIGHLIGHT DETECTION BASELINE")
    print("=" * 60 + "\n")

    # Load optimized config if exists
    config_path = 'configs/optimized_hsv_ranges.json'
    detector = None
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        hsv_ranges = {
            color: {
                'lower': np.array(ranges['lower']),
                'upper': np.array(ranges['upper'])
            }
            for color, ranges in config['hsv_ranges'].items()
        }
        detector = HighlightDetector(
            hsv_ranges=hsv_ranges,
            kernel_size=tuple(config['kernel_size']),
            min_area=config['min_area']
        )
        print(f"Loaded optimized config from {config_path}")
    else:
        detector = HighlightDetector()
        print("Using default detector settings")

    # Image directory
    image_dir = 'data/real/images'
    output_dir = 'outputs/real_baseline'
    os.makedirs(output_dir, exist_ok=True)

    images = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(images)} real samples in {image_dir}\n")

    for img_name in images:
        img_path = os.path.join(image_dir, img_name)
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"  ⚠ Could not load {img_name}")
            continue

        detections = detector.detect(image)
        print(f"  [{img_name}] Detected: {len(detections)} highlights")
        
        for i, det in enumerate(detections):
            print(f"    - {det['color']}: {det['bbox']} conf={det['confidence']:.2f}")

        # Visualize
        vis = detector.visualize_detections(image, detections)
        out_path = os.path.join(output_dir, f"detected_{img_name}")
        cv2.imwrite(out_path, vis)
        print(f"  ✓ Saved visualization to {out_path}\n")

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    test_real_images()
