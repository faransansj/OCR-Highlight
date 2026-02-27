import os
import cv2
import json
import sys
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from unified_pipeline_v2 import UnifiedPipelineV2
from ocr.evaluator import OCREvaluator

def debug_sample(img_idx=4, lang='eng'):
    data_dir = "data/synthetic_v2_large"
    model_path = "final_model/markup_detector_v2_m4.pt"
    
    img_name = f"sample_{img_idx:06d}_{lang}.png"
    img_path = os.path.join(data_dir, img_name)
    json_path = img_path.replace('.png', '.json')
    
    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        return

    with open(json_path, 'r') as f:
        gt_data = json.load(f)
    
    pipeline = UnifiedPipelineV2(model_path=model_path, ocr_engines=['tesseract'])
    evaluator = OCREvaluator()
    
    print(f"Processing {img_name}...")
    result = pipeline.process_image(img_path)
    pred_highlights = result['results']
    gt_highlights = gt_data['annotations']
    
    print(f"GT highlights: {len(gt_highlights)}")
    print(f"Pred highlights: {len(pred_highlights)}")
    
    for i, gt in enumerate(gt_highlights):
        print(f"GT[{i}]: '{gt['text']}' bbox={gt['bbox']}")
    
    for i, p in enumerate(pred_highlights):
        print(f"Pred[{i}]: '{p['text']}' bbox={p['bbox']} conf_det={p['confidence_det']:.2f}")
        
    eval_res = evaluator.evaluate_detections(pred_highlights, gt_highlights)
    print(f"Eval Results:")
    print(f"  Overall CER: {eval_res['overall_cer']:.4f}")
    print(f"  Det Rate: {eval_res['detection_rate']:.2%}")
    
    for m in eval_res['region_metrics']:
        print(f"  Match: GT='{m['ground_truth']}' Pred='{m['predicted']}' CER={m['cer']:.4f}")

    # Visualize
    pipeline.visualize(img_path, result, "outputs/debug_simulation.jpg")
    print(f"Visualization saved to outputs/debug_simulation.jpg")

if __name__ == "__main__":
    debug_sample(4) # sample_000004_eng
