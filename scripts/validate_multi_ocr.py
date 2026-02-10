"""
Multi-engine OCR Validation Script
Validates combined results of MultiOCREngine on synthetic data
"""

import os
import cv2
import json
import numpy as np
import argparse
from tqdm import tqdm
from typing import List, Dict
import logging
from src.ocr.multi_ocr import MultiOCREngine

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCRValidator:
    def __init__(self, data_dir: str, engine_list: List[str] = ['tesseract']):
        self.data_dir = data_dir
        self.engine = MultiOCREngine(default_engines=engine_list)
        
    def validate(self, num_samples: int = 20):
        # Find sample files
        images = [f for f in os.listdir(self.data_dir) if f.endswith('.png')]
        if not images:
            logger.error(f"No images found in {self.data_dir}")
            return
            
        num_samples = min(num_samples, len(images))
        samples = images[:num_samples]
        
        results = []
        
        for img_name in tqdm(samples, desc="Validating OCR"):
            img_path = os.path.join(self.data_dir, img_name)
            json_path = os.path.join(self.data_dir, img_name.replace('.png', '.json'))
            
            if not os.path.exists(json_path):
                continue
                
            # Load image and GT
            image = cv2.imread(img_path)
            with open(json_path, 'r') as f:
                gt_data = json.load(f)
                
            # Run OCR
            ocr_results = self.engine.ensemble_extract(image)
            
            # Simple matching (checking if GT text is present in OCR results)
            gt_texts = [m['text'] for m in gt_data.get('markups', []) if 'text' in m]
            found_count = 0
            
            ocr_full_text = " ".join([res.text for res in ocr_results])
            
            for gt_text in gt_texts:
                if gt_text in ocr_full_text:
                    found_count += 1
            
            results.append({
                "file": img_name,
                "gt_count": len(gt_texts),
                "found_count": found_count,
                "accuracy": found_count / len(gt_texts) if gt_texts else 1.0
            })
            
        # Summary
        total_acc = sum([r['accuracy'] for r in results]) / len(results)
        logger.info(f"Validation Complete. Average Accuracy: {total_acc:.2%}")
        
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/test")
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()
    
    validator = OCRValidator(args.data)
    validator.validate(args.samples)
