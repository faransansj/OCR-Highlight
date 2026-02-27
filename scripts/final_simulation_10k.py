import os
import json
import cv2
import sys
from tqdm import tqdm
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from unified_pipeline_v2 import UnifiedPipelineV2
from ocr.evaluator import OCREvaluator

def run_simulation(limit=100):
    print(f"Starting Final Performance Verification Simulation (Milestone 1)")
    
    data_dir = "data/validation/images"
    model_path = "final_model/markup_detector_v2_real.pt"
    annots_path = "data/validation/validation_annotations.json"
    
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} not found.")
        return

    pipeline = UnifiedPipelineV2(model_path=model_path, ocr_engines=['tesseract'])
    evaluator = OCREvaluator()
    
    with open(annots_path, 'r') as f:
        all_samples = json.load(f)
    
    # Filter for non-augmented
    samples = [s for s in all_samples if not s.get('is_augmented', False)][:limit]
    
    stats = {
        "total_images": 0,
        "total_highlights": 0,
        "correct_detections": 0,
        "total_chars": 0,
        "total_errors": 0
    }
    
    for sample in tqdm(samples, desc="Simulating"):
        img_name = sample['image_name']
        img_path = os.path.join(data_dir, img_name)
        gt_annots = sample.get('annotations', [])
        
        if not os.path.exists(img_path):
            continue

        lang_hint = 'ko' if '_kor' in img_name else 'en'
        if '_jpn' in img_name: lang_hint = 'ja'
        elif '_chi_sim' in img_name: lang_hint = 'zh'

        # 1. Pipeline Prediction
        result = pipeline.process_image(img_path, lang=lang_hint)
        pred_results = result['results']
        
        # 2. Matching
        image_chars = 0
        image_errors = 0
        matched_gt_indices = set()
        
        gt_to_ocr_map = {i: [] for i in range(len(gt_annots))}
        
        for p in pred_results:
            best_iou = 0
            best_gt_idx = -1
            
            for i, gt in enumerate(gt_annots):
                iou = evaluator._calculate_bbox_iou(p['bbox'], gt['bbox'])
                
                # Containment check
                if iou < 0.1:
                    inter = evaluator._calculate_intersection_area(p['bbox'], gt['bbox'])
                    p_area = p['bbox'][2] * p['bbox'][3]
                    if p_area > 0 and (inter / p_area) > 0.5:
                        iou = 0.5
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
            
            if best_gt_idx != -1 and best_iou >= 0.15:
                gt_to_ocr_map[best_gt_idx].append(p)

        for i, gt in enumerate(gt_annots):
            gt_text = gt.get('text', '')
            ref_text = evaluator.normalize_text(gt_text)
            image_chars += len(ref_text)
            
            fragments = gt_to_ocr_map[i]
            if fragments:
                matched_gt_indices.add(i)
                fragments.sort(key=lambda x: x['bbox'][0])
                hyp_text = " ".join([f['text'] for f in fragments])
                metrics = evaluator.calculate_cer(gt_text, hyp_text)
                image_errors += (metrics.insertions + metrics.deletions + metrics.substitutions)
            else:
                image_errors += len(ref_text)

        stats["total_highlights"] += len(gt_annots)
        stats["correct_detections"] += len(matched_gt_indices)
        stats["total_errors"] += image_errors
        stats["total_chars"] += image_chars
        stats["total_images"] += 1

    overall_cer = stats["total_errors"] / stats["total_chars"] if stats["total_chars"] > 0 else 1.0
    det_rate = stats["correct_detections"] / stats["total_highlights"] if stats["total_highlights"] > 0 else 0
    
    print("\n" + "="*50)
    print("FINAL MILESTONE 1 VERIFICATION RESULTS")
    print("="*50)
    print(f"Total Images:      {stats['total_images']}")
    print(f"Detection Rate:    {det_rate:.2%} ({stats['correct_detections']}/{stats['total_highlights']})")
    print(f"OCR CER:           {overall_cer:.4f}")
    print(f"OCR Accuracy:      {max(0, (1-overall_cer)):.2%}")
    print("="*50)
    
    report = {
        "milestone": "Milestone 1 Final Verification",
        "total_samples": stats["total_images"],
        "metrics": {
            "detection_rate": det_rate,
            "cer": overall_cer,
            "accuracy": max(0, 1 - overall_cer)
        }
    }
    with open("outputs/final_10k_simulation_report.json", "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    run_simulation(100)
