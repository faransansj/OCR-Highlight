"""
Advanced Image Preprocessing for OCR
Specialized magic spells for cleaning highlighted regions
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class OCRPreprocessor:
    """Specialized image preprocessing for different highlight colors and markup types"""

    @staticmethod
    def clean_region(image: np.ndarray, color_hint: str = None) -> np.ndarray:
        """
        Clean an image region to make text more visible for OCR
        Uses 'Ultimate Purification Spell': Upscaling + Adaptive Thresholding
        
        Args:
            image: BGR image region
            color_hint: Highlight color ('yellow', 'green', 'pink', 'blue', 'orange')
            
        Returns:
            Cleaned grayscale (or binary) image
        """
        if image is None or image.size == 0:
            return image

        # 1. Rescale (3x) - Tesseract needs high resolution for small snippets
        h, w = image.shape[:2]
        image = cv2.resize(image, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)

        # 2. Add padding (15px)
        image = cv2.copyMakeBorder(image, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        # 3. Convert to grayscale via V channel (HSV)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        
        # 4. Contrast Enhancement (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        v_enhanced = clahe.apply(v)
        
        # 5. Adaptive Thresholding
        return cv2.adaptiveThreshold(v_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 21, 10)

    @staticmethod
    def remove_lines(image: np.ndarray) -> np.ndarray:
        """
        Spell to remove underlines or strikethroughs while keeping text
        Useful when OCR gets confused by horizontal lines
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        
        # Remove lines from original binary
        cleaned_inv = cv2.bitwise_and(binary, cv2.bitwise_not(detected_lines))
        cleaned = cv2.bitwise_not(cleaned_inv)
        
        return cleaned
