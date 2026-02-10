"""
Multi-engine OCR Module
Integrates Tesseract, EasyOCR, and PaddleOCR with specialized preprocessing
"""

import cv2
import numpy as np
import pytesseract
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import logging
import re
from .preprocessor import OCRPreprocessor

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

# Language code mappings for different engines
LANG_MAP = {
    'ko':     {'tesseract': 'kor',     'easyocr': 'ko',     'paddleocr': 'korean'},
    'en':     {'tesseract': 'eng',     'easyocr': 'en',     'paddleocr': 'en'},
    'ja':     {'tesseract': 'jpn',     'easyocr': 'ja',     'paddleocr': 'japan'},
    'zh':     {'tesseract': 'chi_sim', 'easyocr': 'ch_sim', 'paddleocr': 'ch'},
    'ch_sim': {'tesseract': 'chi_sim', 'easyocr': 'ch_sim', 'paddleocr': 'ch'},
}

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
        # Chinese (CJK Unified Ideographs)
        if re.search(r'[\u4e00-\u9fff]', text):
            return 'zh'
        # Default to English if alphanumeric
        if re.search(r'[a-zA-Z]', text):
            return 'en'
        return 'en' # Default to en

    def __init__(
        self,
        default_engines: List[str] = ['easyocr', 'paddleocr'],
        languages: List[str] = ['ko', 'en', 'ja', 'zh'],
        use_gpu: bool = False,
        use_preprocessing: bool = True
    ):
        """
        Initialize Multi-engine OCR
        """
        self.engines = {}
        self.default_engines = default_engines
        self.languages = languages
        self.use_gpu = use_gpu
        self.use_preprocessing = use_preprocessing
        self.preprocessor = OCRPreprocessor()

        # Cache for initialized readers
        self._easyocr_readers = {}
        self._paddle_readers = {}

        # Initialize Tesseract if requested
        if 'tesseract' in default_engines:
            try:
                tess_langs = [LANG_MAP[l]['tesseract'] for l in languages if l in LANG_MAP]
                self.tess_lang_str = '+'.join(tess_langs)
                pytesseract.get_tesseract_version()
                self.engines['tesseract'] = True
            except Exception:
                pass

    def _get_easyocr_reader(self, lang: str):
        # Normalize lang
        lang = 'zh' if lang == 'ch_sim' else lang
        if lang not in self._easyocr_readers:
            try:
                import easyocr
                langs = ['en']
                if lang in LANG_MAP and lang != 'en':
                    langs.append(LANG_MAP[lang]['easyocr'])
                
                logger.info(f"Initializing EasyOCR for {langs}")
                self._easyocr_readers[lang] = easyocr.Reader(langs, gpu=self.use_gpu)
            except Exception as e:
                logger.error(f"Failed to init EasyOCR for {lang}: {e}")
                return None
        return self._easyocr_readers.get(lang)

    def _get_paddle_reader(self, lang: str):
        # Normalize lang
        lang = 'zh' if lang == 'ch_sim' else lang
        if lang not in self._paddle_readers:
            try:
                from paddleocr import PaddleOCR
                paddle_lang = LANG_MAP.get(lang, {}).get('paddleocr', 'korean')
                
                logger.info(f"Initializing PaddleOCR for {paddle_lang}")
                self._paddle_readers[lang] = PaddleOCR(use_angle_cls=True, lang=paddle_lang)
            except Exception as e:
                logger.error(f"Failed to init PaddleOCR for {lang}: {e}")
                return None
        return self._paddle_readers.get(lang)

    def extract_text(
        self,
        image: np.ndarray,
        engine: Optional[str] = None,
        lang: Optional[str] = 'ko',
        color_hint: Optional[str] = None
    ) -> List[OCRResult]:
        """
        Extract text using specified engine
        """
        engine = engine or self.default_engines[0]
        lang = lang or 'ko'
        
        # Apply preprocessing magic
        if self.use_preprocessing:
            processed = self.preprocessor.clean_region(image, color_hint)
            # If preprocessor returned grayscale but engine needs BGR (EasyOCR/PaddleOCR sometimes prefer 3ch)
            if len(processed.shape) == 2 and engine != 'tesseract':
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        else:
            processed = image

        results = []

        if engine == 'tesseract':
            try:
                data = pytesseract.image_to_data(
                    processed, 
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
                            language=lang
                        ))
            except Exception:
                pass

        elif engine == 'easyocr':
            reader = self._get_easyocr_reader(lang)
            if reader:
                try:
                    raw_results = reader.readtext(processed)
                    for (bbox, text, conf) in raw_results:
                        x_min = int(min([p[0] for p in bbox]))
                        y_min = int(min([p[1] for p in bbox]))
                        x_max = int(max([p[0] for p in bbox]))
                        y_max = int(max([p[1] for p in bbox]))
                        results.append(OCRResult(
                            text=text,
                            confidence=float(conf),
                            bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                            engine='easyocr',
                            language=lang
                        ))
                except Exception as e:
                    logger.error(f"EasyOCR error: {e}")

        elif engine == 'paddleocr':
            reader = self._get_paddle_reader(lang)
            if reader:
                try:
                    raw_results = reader.ocr(processed)
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
                                language=lang
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

    def ensemble_extract(self, 
                        image: np.ndarray, 
                        iou_threshold: float = 0.5, 
                        lang: Optional[str] = None,
                        color_hint: Optional[str] = None) -> List[OCRResult]:
        """
        Run multiple engines and combine results using IoU-based merging and voting
        """
        # First pass: use language hint or default to Korean
        target_lang = lang or 'ko'
        all_raw_results = []
        for eng_name in self.default_engines:
            all_raw_results.extend(self.extract_text(image, engine=eng_name, lang=target_lang, color_hint=color_hint))
        
        # Auto-detect language if not specified and re-run if necessary
        if lang is None and all_raw_results:
            combined_text = ' '.join(r.text for r in all_raw_results)
            detected_lang = self.detect_language(combined_text)
            if detected_lang != target_lang:
                logger.info(f"Re-running OCR with detected language: {detected_lang}")
                all_raw_results = []
                for eng_name in self.default_engines:
                    all_raw_results.extend(self.extract_text(image, engine=eng_name, lang=detected_lang, color_hint=color_hint))
        
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
            
            # Select best text from cluster (Voting based)
            best_res = cluster[0]
            text_votes = {}
            for res in cluster:
                norm_text = res.text.replace(" ", "").lower()
                text_votes[norm_text] = text_votes.get(norm_text, 0) + 1
            
            majority_norm = max(text_votes, key=text_votes.get)
            if text_votes[majority_norm] > 1:
                for res in cluster:
                    if res.text.replace(" ", "").lower() == majority_norm:
                        best_res = res
                        best_res.confidence = min(1.0, best_res.confidence + 0.05 * (text_votes[majority_norm] - 1))
                        break
            
            merged_results.append(best_res)
            
        return merged_results
