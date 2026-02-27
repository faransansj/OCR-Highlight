import os
import sys
import json
import cv2
import time
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

def test_nano_speed():
    print("🚀 Testing Milestone 3 [Pocket Hero] - Nano Model Speed & Accuracy")
    
    # Paths (relative to workspace root)
    nano_model_path = "final_model/markup_detector_v2_nano.pt"
    test_image_path = "data/docvqa_with_markups/marked_docvqa_0014.png"
    
    if not os.path.exists(nano_model_path):
        print(f"Nano model not found at {nano_model_path}!")
        return

    # Initialize Nano Model
    model = YOLO(nano_model_path)
    model.to('cpu')
    
    # Warmup
    img = cv2.imread(str(test_image_path))
    model.predict(img, imgsz=640, verbose=False)
    
    # Speed test
    start_time = time.time()
    results = model.predict(img, imgsz=640, verbose=False)
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"Nano Inference Time: {duration:.4f}s")
    
    # Accuracy check (detection count)
    print(f"Detected {len(results[0].boxes)} markups.")
    
    # Save result
    results[0].save("outputs/nano_test_result.jpg")
    print("Result saved to outputs/nano_test_result.jpg")

if __name__ == "__main__":
    test_nano_speed()
