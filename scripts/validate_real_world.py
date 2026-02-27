import os
import sys
import json
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from ocr.evaluator import OCREvaluator
from unified_pipeline_v2 import UnifiedPipelineV2

def validate_real_model():
    print("🚀 Validating Milestone 2 [Real World Adaptation] Model...")
    
    # Paths (relative to ocr-highlight-v2 directory)
    model_path = "final_model/markup_detector_v2_real.pt"
    val_images_dir = "data/yolo_dataset_real/val/images"
    json_data_dir = "data/docvqa_with_markups"
    
    # Initialize Pipeline
    pipeline = UnifiedPipelineV2(model_path=model_path, ocr_engines=['tesseract'])
    evaluator = OCREvaluator()
    
    # Load val images
    val_images = list(Path(val_images_dir).glob("*.png"))
    print(f"Testing on {len(val_images)} real validation samples...")
    
    stats = {
        "total_images": 0,
        "total_highlights": 0,
        "correct_detections": 0,
        "total_chars": 0,
        "total_errors": 0
    }
    
    for img_path in val_images:
        # Load corresponding JSON (from docvqa_with_markups)
        json_path = Path(json_data_dir) / f"{img_path.stem}.json"
        
        if not json_path.exists():
            continue
            
        with open(json_path, 'r') as f:
            gt_data = json.load(f)
            
        gt_annots = gt_data.get('annotations', [])
        
        try:
            result = pipeline.process_image(str(img_path))
            pred_results = result['results']
            
            # Map predictions to formatted result for evaluator
            pred_formatted = [{'bbox': p['bbox'], 'text': p['text']} for p in pred_results]
            
            # Use many-to-one matching
            eval_res = evaluator.evaluate_detections(pred_formatted, gt_annots)
            
            stats["total_highlights"] += eval_res["num_regions"]
            stats["correct_detections"] += eval_res["matched_regions"]
            stats["total_errors"] += eval_res["total_errors"]
            stats["total_chars"] += eval_res["total_characters"]
            stats["total_images"] += 1
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            continue

    overall_cer = stats["total_errors"] / stats["total_chars"] if stats["total_chars"] > 0 else 1.0
    det_rate = stats["correct_detections"] / stats["total_highlights"] if stats["total_highlights"] > 0 else 0
    
    print("\n" + "="*50)
    print("REAL WORLD VALIDATION RESULTS")
    print("="*50)
    print(f"Detection Rate: {det_rate:.2%} ({stats['correct_detections']}/{stats['total_highlights']})")
    print(f"OCR CER:        {overall_cer:.4f}")
    print(f"Accuracy:       {max(0, (1-overall_cer)):.2%}")
    print("="*50)

if __name__ == "__main__":
    validate_real_model()
