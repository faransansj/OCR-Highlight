import numpy as np
from typing import List, Dict, Tuple

def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """Calculate Intersection over Union (IoU) of two bounding boxes [x, y, w, h]"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter_area == 0:
        return 0.0
        
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area

def calculate_intersection_ratio(target_box: List[int], text_box: List[int]) -> float:
    """Calculate what percentage of the text_box is covered by the target_box (highlight/markup)"""
    x1, y1, w1, h1 = target_box
    x2, y2, w2, h2 = text_box
    
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter_area == 0:
        return 0.0
        
    text_box_area = w2 * h2
    return inter_area / text_box_area if text_box_area > 0 else 0.0

def fuse_markup_with_text(markups: List[Dict], ocr_results: List[Dict], threshold: float = 0.5) -> List[Dict]:
    """
    Match markup detections (YOLO) with full-page OCR results.
    
    Args:
        markups: List of dicts with 'bbox' [x, y, w, h], 'label', 'conf'
        ocr_results: List of dicts with 'bbox' [x, y, w, h], 'text', 'conf'
        threshold: Minimum intersection ratio to consider a match
    """
    fused_results = []
    
    for markup in markups:
        matched_texts = []
        
        for ocr in ocr_results:
            # Check how much of the OCR text box is covered by the markup box
            ratio = calculate_intersection_ratio(markup['bbox'], ocr['bbox'])
            
            if ratio >= threshold:
                matched_texts.append({
                    'text': ocr['text'],
                    'ocr_bbox': ocr['bbox'],
                    'ocr_conf': ocr['conf'],
                    'overlap_ratio': ratio
                })
        
        # Sort matched texts roughly by reading order (top-to-bottom, left-to-right)
        matched_texts.sort(key=lambda x: (x['ocr_bbox'][1] // 10, x['ocr_bbox'][0]))
        
        combined_text = " ".join([m['text'] for m in matched_texts])
        avg_ocr_conf = np.mean([m['ocr_conf'] for m in matched_texts]) if matched_texts else 0.0
        
        fused_results.append({
            'markup_type': markup['label'],
            'markup_bbox': markup['bbox'],
            'markup_conf': markup['conf'],
            'text': combined_text,
            'ocr_conf': avg_ocr_conf,
            'matched_fragments': len(matched_texts)
        })
        
    return fused_results
