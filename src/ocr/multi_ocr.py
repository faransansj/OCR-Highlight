"""
Multi-engine OCR Module
Integrates Tesseract, EasyOCR, and PaddleOCR with specialized preprocessing and ensemble logic
"""

import cv2
import numpy as np
import pytesseract
from typing import List, Dict, Optional, Tuple, Union
import logging
from .types import OCRResult
from .preprocessor import OCRPreprocessor
from .ensemble import WeightedEnsemble

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Language code mappings for different engines
LANG_MAP = {
    'ko':     {'tesseract': 'kor',     'easyocr': 'ko',     'paddleocr': 'korean'},
    'en':     {'tesseract': 'eng',     'easyocr': 'en',     'paddleocr': 'en'},
    'ja':     {'tesseract': 'jpn',     'easyocr': 'ja',     'paddleocr': 'japan'},
    'zh':     {'tesseract': 'chi_sim', 'easyocr': 'ch_sim', 'paddleocr': 'ch'},
    'ch_sim': {'tesseract': 'chi_sim', 'easyocr': 'ch_sim', 'paddleocr': 'ch'},
}

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
        # Chinese (CJK Unified Ideographs)
        if re.search(r'[\u4e00-\u9fff]', text):
            return 'zh'
        # Default to English if alphanumeric
        if re.search(r'[a-zA-Z]', text):
            return 'en'
        return 'en'

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
        self.ensemble_manager = WeightedEnsemble()

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
    
    def _post_process_text(self, text: str) -> str:
        """Apply aggressive Korean post-processing"""
        # Remove noise characters at start/end
        # Specifically target small noise characters often misidentified as brackets or dots
        text = re.sub(r'^[\W_]+|[\W_]+$', '', text, flags=re.UNICODE)
        
        # Remove common standalone noise characters at the beginning (e.g. '(', '.', '|', '/')
        text = re.sub(r'^[()|/.,:;]+', '', text)
        
        # Aggressive Korean space removal
        prev_text = None
        while prev_text != text:
            prev_text = text
            text = re.sub(r'([\uac00-\ud7af])\s+([\uac00-\ud7af])', r'\1\2', text)
        
        # Also remove spaces after Korean before particles
        text = re.sub(r'([\uac00-\ud7af])\s+([은는이가을를에서])\b', r'\1\2', text)
        
        # Fix common duplications
        text = re.sub(r'([\uac00-\ud7af]{2,})\s*([은를을이가])\1\s*\2', r'\1\2', text)
        
        return text

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
            # If preprocessor returned grayscale but engine needs BGR
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
                    config='--psm 7',
                    output_type=pytesseract.Output.DICT
                )
                for i in range(len(data['text'])):
                    text = data['text'][i].strip()
                    conf = float(data['conf'][i])
                    if text and conf > 0:
                        text = self._post_process_text(text)
                        if not text: continue
                        
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
                        iou_threshold: float = 0.3, # Lowered from 0.5 to catch more overlaps
                        lang: Optional[str] = None,
                        color_hint: Optional[str] = None) -> List[OCRResult]:
        """
        Run multiple engines and combine results using IoU-based merging and weighted voting
        """
        target_lang = lang or 'ko'
        all_raw_results = []
        for eng_name in self.default_engines:
            all_raw_results.extend(self.extract_text(image, engine=eng_name, lang=target_lang, color_hint=color_hint))
        
        if lang is None and all_raw_results:
            combined_text = ' '.join(r.text for r in all_raw_results)
            detected_lang = self.detect_language(combined_text)
            if detected_lang != target_lang:
                logger.info(f"Re-running OCR with detected language: {detected_lang}")
                all_raw_results = []
                for eng_name in self.default_engines:
                    all_raw_results.extend(self.extract_text(image, engine=eng_name, lang=detected_lang, color_hint=color_hint))
                target_lang = detected_lang
        
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
                
                res1 = all_raw_results[i]
                res2 = all_raw_results[j]
                iou = self._calculate_iou(res1.bbox, res2.bbox)
                
                # Check for duplication: high IoU OR same text in very close proximity
                is_duplicate = iou > iou_threshold
                
                # Aggressive string similarity check for duplication in small regions
                if not is_duplicate and iou >= 0:
                    text1 = res1.text.replace(" ", "").lower()
                    text2 = res2.text.replace(" ", "").lower()
                    
                    # 1. Substring match + any spatial overlap
                    if len(text1) > 1 and len(text2) > 1:
                        if text1 in text2 or text2 in text1:
                            is_duplicate = True
                    
                    # 2. High character overlap check + spatial proximity
                    if not is_duplicate and len(text1) >= 2 and len(text2) >= 2:
                        chars1 = set(text1)
                        chars2 = set(text2)
                        intersection = chars1.intersection(chars2)
                        overlap_ratio = len(intersection) / max(len(chars1), len(chars2))
                        if overlap_ratio > 0.6: # Relaxed from 0.7
                            is_duplicate = True
                    
                    # 3. Horizontal alignment (Same line/word detection)
                    if not is_duplicate:
                        y_overlap = max(0, min(res1.bbox[1]+res1.bbox[3], res2.bbox[1]+res2.bbox[3]) - max(res1.bbox[1], res2.bbox[1]))
                        h_min = min(res1.bbox[3], res2.bbox[3])
                        # If y-overlap is high and they are close horizontally
                        if y_overlap > 0.7 * h_min:
                            dist = max(0, max(res1.bbox[0], res2.bbox[0]) - min(res1.bbox[0]+res1.bbox[2], res2.bbox[0]+res2.bbox[2]))
                            if dist < 5: # Very close horizontally
                                is_duplicate = True

                if is_duplicate:
                    cluster.append(res2)
                    used_indices.add(j)
            
            # Weighted selection from cluster
            # First, multiply confidence by engine-language weight
            for res in cluster:
                weight = self.ensemble_manager.WEIGHTS.get(target_lang, {}).get(res.engine, 1.0)
                res.confidence *= weight
            
            # Sort cluster again by weighted confidence
            cluster.sort(key=lambda x: x.confidence, reverse=True)
            best_res = cluster[0]
            
            # Voting agreement bonus
            text_votes = {}
            for res in cluster:
                norm_text = res.text.replace(" ", "").lower()
                text_votes[norm_text] = text_votes.get(norm_text, 0) + 1
            
            majority_norm = max(text_votes, key=text_votes.get)
            
            # If there's a majority vote, use that text instead of just highest confidence
            if text_votes[majority_norm] > 1:
                best_res.confidence = min(2.0, best_res.confidence + 0.1)
                # Find the representative text for this majority
                for res in cluster:
                    if res.text.replace(" ", "").lower() == majority_norm:
                        best_res.text = res.text
                        break
            else:
                # If no majority, and highest confidence has noise but others don't, 
                # prefer the cleaner one if confidence is close
                clean_res = None
                for res in cluster:
                    if res == best_res: continue
                    is_clean = not re.search(r'[^\w\s\uac00-\ud7af]', res.text)
                    if is_clean and res.confidence > 0.8 * best_res.confidence:
                        clean_res = res
                        break
                if clean_res:
                    best_res.text = clean_res.text
                    best_res.confidence = (best_res.confidence + clean_res.confidence) / 2
            
            merged_results.append(best_res)
            
        return merged_results

if __name__ == "__main__":
    # Quick test
    engine = MultiOCREngine(default_engines=['easyocr'], languages=['ko', 'en'])
    test_img = np.ones((100, 300, 3), dtype=np.uint8) * 255
    cv2.putText(test_img, "Test Text", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    
    res = engine.ensemble_extract(test_img)
    print(f"Detected: {res}")
