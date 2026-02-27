from dataclasses import dataclass
from typing import List

@dataclass
class OCRResult:
    """OCR result for a single region"""
    text: str
    confidence: float
    bbox: List[int]  # [x, y, w, h]
    engine: str
    language: str
    color: str = "unknown"
