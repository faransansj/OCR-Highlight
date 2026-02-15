"""
Unified Markup Detection & OCR Pipeline v2.0
Integrates YOLOv8 (Large) with Multi-engine Ensemble OCR and specialized preprocessing
"""

import os
import cv2
import numpy as np
import json
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from ultralytics import YOLO
from src.ocr.multi_ocr import MultiOCREngine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UnifiedResultV2:
    """Unified result for detection + OCR"""
    markup_type: str
    subtype: str
    bbox: List[int] # [x, y, w, h]
    confidence_det: float
    text: str
    confidence_ocr: float
    engine: str
    color: Optional[str] = None

class UnifiedPipelineV2:
    """
    High-performance pipeline for Milestone 1
    Uses YOLOv8l for detection and MultiOCREngine for extraction
    """
    
    def __init__(self, 
                 model_path: str = "final_model/markup_detector_v1.pt",
                 ocr_engines: List[str] = ['easyocr', 'paddleocr'],
                 use_gpu: bool = False):
        """
        Initialize Pipeline
        """
        logger.info(f"Initializing Pipeline v2.0 with model: {model_path}")
        self.detector = YOLO(model_path)
        self.ocr_engine = MultiOCREngine(default_engines=ocr_engines, use_gpu=use_gpu)
        
        # Color mapping for visualization
        self.color_palette = {
            'highlight': (0, 255, 255),
            'underline': (0, 255, 0),
            'strikethrough': (0, 0, 255),
            'circle': (255, 0, 0),
            'rectangle': (255, 128, 0)
        }

    def process_image(self, image_path: str, lang: Optional[str] = None) -> Dict:
        """
        Full process: Detection -> Preprocessing -> OCR -> Ensemble
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found: {image_path}")

        # 1. Detection Phase (YOLOv8)
        # Using imgsz=1280 for better small markup detection
        results_det = self.detector.predict(image, imgsz=1280, conf=0.25)
        
        final_results = []
        
        for result in results_det:
            boxes = result.boxes
            for i in range(len(boxes)):
                box = boxes[i]
                cls = int(box.cls[0])
                label = self.detector.names[cls]
                conf_det = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                
                # Convert to [x, y, w, h]
                x, y, x2, y2 = map(int, xyxy)
                w, h = x2 - x, y2 - y
                bbox = [x, y, w, h]
                
                # 2. Preprocessing & OCR Phase
                # For highlights, we can try to detect color
                color_hint = None
                if label == 'highlight':
                    color_hint = self._detect_color(image[y:y2, x:x2])
                
                # Run OCR Ensemble
                ocr_outputs = self.ocr_engine.ensemble_extract(
                    image, 
                    iou_threshold=0.5, 
                    lang=lang,
                    color_hint=color_hint
                )
                
                # Pick the best text (currently first in ensemble result)
                best_text = ""
                conf_ocr = 0.0
                engine_name = "none"
                
                if ocr_outputs:
                    # Search for result overlapping with this bbox
                    # Or just use the one from the crop if we cropped (current ensemble_extract takes full image)
                    # For performance, we should ideally crop first.
                    # Refactoring MultiOCR to take regions more efficiently:
                    region = image[max(0,y-5):min(image.shape[0],y2+5), max(0,x-5):min(image.shape[1],x2+5)]
                    region_results = self.ocr_engine.ensemble_extract(region, lang=lang, color_hint=color_hint)
                    
                    if region_results:
                        best_res = region_results[0]
                        best_text = best_res.text
                        conf_ocr = best_res.confidence
                        engine_name = best_res.engine

                final_results.append(UnifiedResultV2(
                    markup_type=label,
                    subtype=color_hint or "standard",
                    bbox=bbox,
                    confidence_det=conf_det,
                    text=best_text,
                    confidence_ocr=conf_ocr,
                    engine=engine_name,
                    color=color_hint
                ))

        return {
            "image_path": image_path,
            "results": [asdict(r) for r in final_results]
        }

    def _detect_color(self, region: np.ndarray) -> str:
        """Detect dominant highlight color in BGR region"""
        if region.size == 0: return "unknown"
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        h = hsv[:,:,0]
        avg_h = np.median(h)
        
        if 20 <= avg_h <= 40: return "yellow"
        if 40 < avg_h <= 80: return "green"
        if 140 <= avg_h <= 170: return "pink"
        return "unknown"

    def visualize(self, image_path: str, results: Dict, output_path: str):
        """Save annotated image"""
        img = cv2.imread(image_path)
        for res in results['results']:
            x, y, w, h = res['bbox']
            color = self.color_palette.get(res['markup_type'], (255, 255, 255))
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            
            label = f"{res['markup_type']}: {res['text']}"
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        cv2.imwrite(output_path, img)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model", type=str, default="final_model/markup_detector_v1.pt")
    args = parser.parse_args()
    
    pipeline = UnifiedPipelineV2(model_path=args.model)
    res = pipeline.process_image(args.image)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    
    pipeline.visualize(args.image, res, "output_v2.jpg")
