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
from src.fusion import fuse_markup_with_text

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
    High-performance pipeline for Milestone 1 & 5
    Uses YOLOv8 for detection and MultiOCREngine/PaddleOCR-VL for extraction
    """
    
    def __init__(self, 
                 model_path: str = "final_model/markup_detector_v2_nano.pt",
                 ocr_engines: List[str] = ['easyocr', 'tesseract'],
                 use_gpu: bool = False,
                 use_vlm: bool = False,
                 use_remote_vlm: bool = False,
                 vlm_server_url: str = "http://localhost:8000/v1/ocr"):
        """
        Initialize Pipeline
        """
        logger.info(f"Initializing Pipeline v2.0 with model: {model_path}")
        self.detector = YOLO(model_path)
        self.use_vlm = use_vlm or use_remote_vlm
        
        if use_remote_vlm:
            from src.ocr.vlm_client import PaddleOCRVLClient
            self.vlm_engine = PaddleOCRVLClient(api_url=vlm_server_url)
            logger.info(f"Remote VLM Client (PaddleOCR-VL-1.5) enabled via {vlm_server_url}.")
        elif use_vlm:
            from src.ocr.paddleocr_vl import PaddleOCRVL
            self.vlm_engine = PaddleOCRVL(use_gpu=use_gpu)
            logger.info("Local VLM Engine (PaddleOCR-VL-1.5) enabled for fusion mode.")
        else:
            self.ocr_engine = MultiOCREngine(default_engines=ocr_engines, use_gpu=use_gpu)
            logger.info(f"Standard OCR Engines enabled: {ocr_engines}")
        
        # Color mapping for visualization
        self.color_palette = {
            'highlight': (0, 255, 255),
            'underline': (0, 255, 0),
            'strikethrough': (0, 0, 255),
            'circle': (255, 0, 0),
            'rectangle': (255, 128, 0)
        }

    def enrich_metadata(self, results: List[Dict]) -> str:
        """
        Convert structured results into enriched Markdown text.
        Tags texts with corresponding markup types and colors.
        """
        enriched_lines = []
        for res in results:
            text = res.get('text', '')
            if not text:
                continue
                
            m_type = res.get('markup_type', '')
            subtype = res.get('subtype', 'standard')
            
            if m_type == 'highlight':
                color_map = {'yellow': '#ffff00', 'green': '#00ff00', 'pink': '#ff00ff', 'unknown': '#ffffcc'}
                color_hex = color_map.get(subtype, '#ffffcc')
                enriched = f'<mark style="background-color: {color_hex}">{text}</mark>'
            elif m_type == 'underline':
                enriched = f'<u>{text}</u>'
            elif m_type == 'strikethrough':
                enriched = f'<s>{text}</s>'
            elif m_type == 'circle':
                enriched = f'<span style="border: 2px solid red; border-radius: 50%; padding: 2px;">{text}</span>'
            elif m_type == 'rectangle':
                enriched = f'<span style="border: 2px solid blue; padding: 2px;">{text}</span>'
            else:
                enriched = text
                
            enriched_lines.append(enriched)
            
        return "\n\n".join(enriched_lines)

    def process_image(self, image_path: str, lang: Optional[str] = None, mode: str = "crop") -> Dict:
        """
        Full process: Detection -> Preprocessing -> OCR -> Ensemble
        mode: 'crop' (legacy fallback) or 'fusion' (fast full-page matching)
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found: {image_path}")

        # 1. Detection Phase (YOLOv8)
        # Using imgsz=1280 for better small markup detection
        results_det = self.detector.predict(image, imgsz=1280, conf=0.10)
        
        final_results = []
        
        if mode == "fusion":
            # 2. Extract ALL text from the full page once
            if self.use_vlm:
                ocr_dicts = self.vlm_engine.extract_text(image, lang=lang)
            else:
                full_page_ocr = self.ocr_engine.ensemble_extract(image, lang=lang)
                ocr_dicts = [{'bbox': r.bbox, 'text': r.text, 'conf': r.confidence} for r in full_page_ocr]
            
            # 3. Gather markups
            markups = []
            for result in results_det:
                boxes = result.boxes
                for i in range(len(boxes)):
                    box = boxes[i]
                    cls = int(box.cls[0])
                    label = self.detector.names[cls]
                    conf_det = float(box.conf[0])
                    x, y, x2, y2 = map(int, box.xyxy[0].tolist())
                    markups.append({'bbox': [x, y, x2 - x, y2 - y], 'label': label, 'conf': conf_det})
            
            # 4. Fuse markup with text boxes
            fused = fuse_markup_with_text(markups, ocr_dicts)
            
            for f in fused:
                color_hint = None
                if f['markup_type'] == 'highlight':
                    x, y, w, h = f['markup_bbox']
                    color_hint = self._detect_color(image[max(0, y):y+h, max(0, x):x+w])
                    
                final_results.append(UnifiedResultV2(
                    markup_type=f['markup_type'],
                    subtype=color_hint or "standard",
                    bbox=f['markup_bbox'],
                    confidence_det=f['markup_conf'],
                    text=f['text'],
                    confidence_ocr=f['ocr_conf'],
                    engine="fusion",
                    color=color_hint
                ))
                
        else:
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
                    
                    # Crop region with small margin for better OCR context
                    margin = 5
                    ry1, ry2 = max(0, y-margin), min(image.shape[0], y2+margin)
                    rx1, rx2 = max(0, x-margin), min(image.shape[1], x2+margin)
                    region = image[ry1:ry2, rx1:rx2]
    
                    # Run OCR Ensemble on the region
                    best_text = ""
                    conf_ocr = 0.0
                    engine_name = "none"
    
                    if region.size > 0:
                        region_results = self.ocr_engine.ensemble_extract(
                            region, 
                            lang=lang, 
                            color_hint=color_hint
                        )
                        
                        if region_results:
                            # Join all detected fragments in the region
                            # Sort by x coordinate first
                            region_results.sort(key=lambda r: r.bbox[0])
                            best_text = " ".join([r.text for r in region_results])
                            conf_ocr = np.mean([r.confidence for r in region_results])
                            engine_name = region_results[0].engine
    
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
            "markdown": self.enrich_metadata([asdict(r) for r in final_results]),
            "results": [asdict(r) for r in final_results]
        }

    def export_results(self, result_dict: Dict, output_base: str):
        """Export results to JSON and Markdown files"""
        # Save JSON
        with open(f"{output_base}.json", "w", encoding="utf-8") as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)
            
        # Save Markdown
        with open(f"{output_base}.md", "w", encoding="utf-8") as f:
            f.write(f"# OCR Extraction Results\n\n")
            f.write(f"**Source**: `{result_dict['image_path']}`\n\n")
            f.write("## Extracted Content\n\n")
            f.write(result_dict.get('markdown', ''))
            
        logger.info(f"Exported results to {output_base}.json and {output_base}.md")

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
