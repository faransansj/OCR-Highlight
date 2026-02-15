"""
Weighted Voting Ensemble for OCR
Assigns weights to different engines based on language performance
"""

import logging
from typing import List, Dict
from .multi_ocr import OCRResult

logger = logging.getLogger(__name__)

class WeightedEnsemble:
    """Ensemble logic with engine-language weights"""
    
    # Weights for each engine by language (Performance heuristics)
    # Higher is better
    WEIGHTS = {
        'ko': {
            'paddleocr': 1.2,
            'easyocr': 1.0,
            'tesseract': 0.7
        },
        'ja': {
            'paddleocr': 1.2,
            'easyocr': 1.1,
            'tesseract': 0.6
        },
        'en': {
            'easyocr': 1.2,
            'tesseract': 1.1,
            'paddleocr': 1.0
        },
        'zh': {
            'paddleocr': 1.3,
            'easyocr': 1.0,
            'tesseract': 0.7
        }
    }

    def merge(self, results: List[OCRResult], language: str) -> List[OCRResult]:
        """
        Merge results using weighted voting based on language
        """
        if not results:
            return []
            
        # Group by IoU clusters (already done in ensemble_extract usually)
        # But here we focus on picking the best from a cluster using weights
        
        # Implementation of weighted selection logic
        # If multiple results exist for one region, we multiply confidence by language weight
        for res in results:
            engine_weight = self.WEIGHTS.get(language, {}).get(res.engine, 1.0)
            res.confidence *= engine_weight
            
        return results
