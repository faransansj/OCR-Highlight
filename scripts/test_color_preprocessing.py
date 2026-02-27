import sys
import os
import cv2
import json
import numpy as np
from typing import List, Dict

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from ocr.preprocessor import OCRPreprocessor
import pytesseract

def lev_distance(s1, s2):
    if len(s1) < len(s2):
        return lev_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def calculate_cer(gt: str, pred: str) -> float:
    if not gt: return 0.0 if not pred else 1.0
    return lev_distance(gt, pred) / len(gt)

def test_color_preprocessing(num_samples: int = 30):
    print("Testing Preprocessing Variants (K-means + Closing)...")
    
    with open('data/validation/validation_annotations.json', 'r') as f:
        val_data = json.load(f)
    
    val_data = [d for d in val_data if not d.get('is_augmented', False)]
    
    results = []
    
    for sample in val_data[:num_samples]:
        img_path = os.path.join('data/validation/images', sample['image_name'])
        image = cv2.imread(img_path)
        if image is None: continue
        
        for highlight in sample.get('highlight_annotations', []):
            color = highlight['color']
            gt_text = highlight['text']
            bbox = [int(v) for v in highlight['bbox']]
            x, y, w, h = bbox
            
            # Crop region
            crop = image[y:y+h, x:x+w]
            if crop.size == 0: continue
            
            # Variant: Cleaned (K-means + Closing logic inside clean_region)
            proc_cleaned = OCRPreprocessor.clean_region(crop, color_hint=color)
            text_cleaned = pytesseract.image_to_string(proc_cleaned, lang='kor+eng', config='--psm 7').strip()
            cer_cleaned = calculate_cer(gt_text, text_cleaned)
            
            # Base (Simple Gray + Otsu, No Upscale)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, proc_base = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text_base = pytesseract.image_to_string(proc_base, lang='kor+eng', config='--psm 7').strip()
            cer_base = calculate_cer(gt_text, text_base)

            results.append({
                'color': color,
                'gt': gt_text,
                'base': text_base,
                'cleaned': text_cleaned,
                'cer_base': cer_base,
                'cer_cleaned': cer_cleaned
            })

    # Summary
    avg_cer_base = sum(r['cer_base'] for r in results) / len(results)
    avg_cer_cleaned = sum(r['cer_cleaned'] for r in results) / len(results)
    
    print(f"\nResults over {len(results)} highlights:")
    print(f"Average CER (Base - No Upscale): {avg_cer_base:.4f}")
    print(f"Average CER (Cleaned - Magic):    {avg_cer_cleaned:.4f}")
    
    # Show some examples
    print("\nExamples:")
    for r in results[:10]:
        print(f"[{r['color']}] GT: '{r['gt']}'")
        print(f"  Base: '{r['base']}' (CER: {r['cer_base']:.3f})")
        print(f"  Cln:  '{r['cleaned']}' (CER: {r['cer_cleaned']:.3f})")

if __name__ == "__main__":
    test_color_preprocessing(30)
