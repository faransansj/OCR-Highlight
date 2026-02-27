"""
Final Performance Verification Simulation (Milestone 1)
Validates UnifiedPipelineV2 on synthetic dataset
"""

import os
import json
import cv2
import logging
from typing import List, Dict
from src.unified_pipeline_v2 import UnifiedPipelineV2
from src.ocr.evaluator import OCREvaluator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_validation(
    annotations_path: str = "ocr-highlight-v2/data/synthetic/base_annotations.json",
    limit: int = 50,
    output_report: str = "validation_report_v2.json"
):
    # Fix paths for execution context
    base_dir = "ocr-highlight-v2"
    abs_annotations_path = os.path.join(os.getcwd(), annotations_path)
    
    with open(abs_annotations_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    pipeline = UnifiedPipelineV2(model_path=os.path.join(base_dir, "final_model/markup_detector_v1.pt"))
    evaluator = OCREvaluator()
    
    stats = {
        "total_images": 0,
        "total_highlights": 0,
        "correct_detections": 0,
        "total_chars": 0,
        "total_errors": 0,
        "results": []
    }
    
    for i, entry in enumerate(data[:limit]):
        img_name = entry['image_name']
        img_path = os.path.join(base_dir, "data/synthetic", img_name)
        gt_highlights = entry['highlight_annotations']
        
        logger.info(f"[{i+1}/{limit}] Processing {img_name}...")
        
        try:
            pred = pipeline.process_image(img_path)
            # In synthetic data, 'rectangle' often corresponds to GT highlights
            pred_highlights = pred['results']
            
            # Match pred to GT
            matches = evaluator._match_results_to_gt(pred_highlights, gt_highlights, iou_threshold=0.2)
            
            stats["total_highlights"] += len(gt_highlights)
            
            image_errors = 0
            image_chars = 0
            
            for p, gt in matches:
                if gt:
                    stats["correct_detections"] += 1
                    res = evaluator.calculate_cer(gt['text'], p['text'])
                    image_errors += (res.insertions + res.deletions + res.substitutions)
                    image_chars += res.total_chars
                    if stats["correct_detections"] <= 5:
                         logger.info(f"MATCH: GT='{gt['text']}' PRED='{p['text']}' ERRORS={res.insertions + res.deletions + res.substitutions}")
                else:
                    # False positive
                    pass
            
            stats["total_chars"] += image_chars
            stats["total_errors"] += image_errors
            stats["total_images"] += 1
            
        except Exception as e:
            logger.error(f"Error processing {img_name}: {e}")
            continue

    # Summary
    if stats["total_chars"] > 0:
        overall_cer = stats["total_errors"] / stats["total_chars"]
    else:
        overall_cer = 1.0
        
    det_rate = stats["correct_detections"] / stats["total_highlights"] if stats["total_highlights"] > 0 else 0
    
    report = {
        "summary": {
            "overall_cer": overall_cer,
            "detection_rate": det_rate,
            "total_images": stats["total_images"],
            "total_highlights": stats["total_highlights"],
            "correct_detections": stats["correct_detections"],
            "total_chars": stats["total_chars"],
            "total_errors": stats["total_errors"]
        }
    }
    
    with open(output_report, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    print("\n" + "="*40)
    print("VALIDATION REPORT SUMMARY")
    print("="*40)
    print(f"Overall CER: {overall_cer:.4f}")
    print(f"Detection Rate: {det_rate:.2%} ({stats['correct_detections']}/{stats['total_highlights']})")
    print(f"Total Chars: {stats['total_chars']}")
    print(f"Total Errors: {stats['total_errors']}")
    print("="*40)

if __name__ == "__main__":
    run_validation(limit=5)
