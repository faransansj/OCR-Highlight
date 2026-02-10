"""
Multi-engine OCR Module
Integrates Tesseract, EasyOCR, and PaddleOCR
"""

import cv2
import numpy as np
import pytesseract
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OCRResult:
    """OCR result for a single region"""
    text: str
    confidence: float
    bbox: List[int]  # [x, y, w, h]
    engine: str
    language: str

import re

class MultiOCREngine:
    """Multi-engine OCR wrapper supporting Tesseract, EasyOCR, and PaddleOCR"""

    def detect_language(self, text: str) -> str:
        """Detect language based on character sets (simple regex approach)"""
        # Korean (Hangul)
        if re.search(r'[\uac00-\ud7af]', text):
            return 'ko'
        # Japanese (Hiragana/Katakana)
        if re.search(r'[\u3040-\u30ff]', text):
            return 'ja'
        # Chinese (CJK Unified Ideographs - checking for common Han)
        if re.search(r'[\u4e00-\u9fff]', text):
            return 'zh'
        # Default to English if alphanumeric
        if re.search(r'[a-zA-Z]', text):
            return 'en'
        return 'unknown'

    def __init__(
        self,
        default_engines: List[str] = ['tesseract'],
        languages: List[str] = ['ko', 'en'],
        use_gpu: bool = False
    ):
        """
        Initialize Multi-engine OCR

        Args:
            default_engines: List of engines to use ('tesseract', 'easyocr', 'paddleocr')
            languages: List of languages ('ko', 'en', 'ja', 'ch_sim')
            use_gpu: Whether to use GPU for EasyOCR/PaddleOCR
        """
        self.engines = {}
        self.default_engines = default_engines
        self.languages = languages
        self.use_gpu = use_gpu

        # Initialize Tesseract
        if 'tesseract' in default_engines:
            try:
                # Map ISO languages to Tesseract codes
                tess_langs = []
                for lang in languages:
                    if lang == 'ko': tess_langs.append('kor')
                    elif lang == 'en': tess_langs.append('eng')
                    elif lang == 'ja': tess_langs.append('jpn')
                    elif lang == 'ch_sim': tess_langs.append('chi_sim')
                
                self.tess_lang_str = '+'.join(tess_langs)
                pytesseract.get_tesseract_version()
                self.engines['tesseract'] = True
                logger.info(f"Tesseract initialized with langs: {self.tess_lang_str}")
            except Exception as e:
                logger.warning(f"Tesseract not available: {e}")

        # EasyOCR and PaddleOCR will be initialized lazily to avoid heavy imports if not used
        self.easyocr_reader = None
        self.paddle_reader = None

    def _init_easyocr(self):
        if self.easyocr_reader is None:
            try:
                import easyocr
                # Map ISO languages to EasyOCR codes
                easy_langs = []
                for lang in self.languages:
                    if lang == 'ko': easy_langs.append('ko')
                    elif lang == 'en': easy_langs.append('en')
                    elif lang == 'ja': easy_langs.append('ja')
                    elif lang == 'ch_sim': easy_langs.append('ch_sim')
                
                self.easyocr_reader = easyocr.Reader(easy_langs, gpu=self.use_gpu)
                self.engines['easyocr'] = True
                logger.info(f"EasyOCR initialized with langs: {easy_langs}")
            except ImportError:
                logger.warning("EasyOCR not installed. Run 'pip install easyocr'")
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR: {e}")

    def _init_paddleocr(self):
        if self.paddle_reader is None:
            try:
                from paddleocr import PaddleOCR
                # Map ISO languages to PaddleOCR codes
                # Note: PaddleOCR usually handles one language at a time or multi-lingual model
                self.paddle_reader = PaddleOCR(use_angle_cls=True, lang='korean', use_gpu=self.use_gpu)
                self.engines['paddleocr'] = True
                logger.info("PaddleOCR initialized")
            except ImportError:
                logger.warning("PaddleOCR not installed. Run 'pip install paddleocr paddlepaddle'")
            except Exception as e:
                logger.warning(f"Failed to initialize PaddleOCR: {e}")

    def extract_text(
        self,
        image: np.ndarray,
        engine: Optional[str] = None,
        lang: Optional[str] = None
    ) -> List[OCRResult]:
        """
        Extract text using specified engine

        Args:
            image: Input image (BGR)
            engine: Engine to use (overrides default)
            lang: Language hint

        Returns:
            List of OCRResult objects
        """
        engine = engine or self.default_engines[0]
        results = []

        if engine == 'tesseract':
            try:
                # PSM 6: Assume a single uniform block of text
                data = pytesseract.image_to_data(
                    image, 
                    lang=self.tess_lang_str, 
                    config='--psm 6',
                    output_type=pytesseract.Output.DICT
                )
                
                for i in range(len(data['text'])):
                    text = data['text'][i].strip()
                    conf = float(data['conf'][i])
                    if text and conf > 0:
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        results.append(OCRResult(
                            text=text,
                            confidence=conf/100.0,
                            bbox=[x, y, w, h],
                            engine='tesseract',
                            language=lang or 'unknown'
                        ))
            except Exception as e:
                logger.error(f"Tesseract error: {e}")

        elif engine == 'easyocr':
            self._init_easyocr()
            if self.easyocr_reader:
                try:
                    # EasyOCR returns list of (bbox, text, confidence)
                    # bbox is [[x,y], [x,y], [x,y], [x,y]]
                    raw_results = self.easyocr_reader.readtext(image)
                    for (bbox, text, conf) in raw_results:
                        # Convert bbox to [x, y, w, h]
                        x_min = int(min([p[0] for p in bbox]))
                        y_min = int(min([p[1] for p in bbox]))
                        x_max = int(max([p[0] for p in bbox]))
                        y_max = int(max([p[1] for p in bbox]))
                        results.append(OCRResult(
                            text=text,
                            confidence=float(conf),
                            bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                            engine='easyocr',
                            language=lang or 'unknown'
                        ))
                except Exception as e:
                    logger.error(f"EasyOCR error: {e}")

        elif engine == 'paddleocr':
            self._init_paddleocr()
            if self.paddle_reader:
                try:
                    # PaddleOCR returns list of [bbox, (text, confidence)]
                    raw_results = self.paddle_reader.ocr(image, cls=True)
                    if raw_results and raw_results[0]:
                        for line in raw_results[0]:
                            bbox, (text, conf) = line
                            x_min = int(min([p[0] for p in bbox]))
                            y_min = int(min([p[1] for p in bbox]))
                            x_max = int(max([p[0] for p in bbox]))
                            y_max = int(max([p[1] for p in bbox]))
                            results.append(OCRResult(
                                text=text,
                                confidence=float(conf),
                                bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                                engine='paddleocr',
                                language=lang or 'unknown'
                            ))
                except Exception as e:
                    logger.error(f"PaddleOCR error: {e}")

        return results

    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate Intersection over Union of two bboxes [x, y, w, h]"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union_area = (w1 * h1) + (w2 * h2) - inter_area
        
        return inter_area / union_area if union_area > 0 else 0

    def ensemble_extract(self, image: np.ndarray, iou_threshold: float = 0.5) -> List[OCRResult]:
        """
        Run multiple engines and combine results using IoU-based merging and voting
        
        Args:
            image: Input image
            iou_threshold: IoU threshold for merging boxes
            
        Returns:
            Merged OCR results
        """
        all_raw_results = []
        for eng_name in self.default_engines:
            all_raw_results.extend(self.extract_text(image, engine=eng_name))
        
        if not all_raw_results:
            return []

        # Sort by confidence descending
        all_raw_results.sort(key=lambda x: x.confidence, reverse=True)
        
        merged_results = []
        used_indices = set()
        
        for i in range(len(all_raw_results)):
            if i in used_indices:
                continue
            
            cluster = [all_raw_results[i]]
            used_indices.add(i)
            
            for j in range(i + 1, len(all_raw_results)):
                if j in used_indices:
                    continue
                
                if self._calculate_iou(all_raw_results[i].bbox, all_raw_results[j].bbox) > iou_threshold:
                    cluster.append(all_raw_results[j])
                    used_indices.add(j)
            
            # Select best text from cluster (Majority vote or highest confidence)
            # For now, highest confidence (first in sorted cluster)
            best_res = cluster[0]
            
            # If multiple engines agree on the same text, boost confidence
            text_votes = {}
            for res in cluster:
                text_votes[res.text] = text_votes.get(res.text, 0) + 1
            
            # Find most frequent text
            majority_text = max(text_votes, key=text_votes.get)
            if text_votes[majority_text] > 1:
                # Use majority text if agreement exists
                for res in cluster:
                    if res.text == majority_text:
                        best_res = res
                        # Boost confidence slightly for agreement
                        best_res.confidence = min(1.0, best_res.confidence + 0.05 * (text_votes[majority_text] - 1))
                        break
            
            merged_results.append(best_res)
            
        return merged_results

if __name__ == "__main__":
    # Quick test
    engine = MultiOCREngine(default_engines=['tesseract'], languages=['ko', 'en'])
    test_img = np.ones((100, 300, 3), dtype=np.uint8) * 255
    cv2.putText(test_img, "Test Text", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    
    res = engine.extract_text(test_img)
    print(f"Detected: {res}")
