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
        
        Args:
            image: BGR image region
            color_hint: Highlight color ('yellow', 'green', 'pink', 'blue', 'orange')
            
        Returns:
            Cleaned grayscale (or binary) image
        """
        if image is None or image.size == 0:
            return image

        # 1. Basic Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 2. Targeted color removal using HLS/HSV (The "Purify" spell)
        if color_hint:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # Special logic for problematic colors
            if color_hint in ['pink', 'red']:
                # Pink is tough. The text is black, background is light red/pink.
                # Use adaptive threshold on Value channel with high block size
                # This helps preserve the "blackness" of text regardless of background
                return cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 21, 10)
                
            elif color_hint == 'green':
                # Green sometimes overlaps with black in grayscale.
                # Using the V channel from HSV usually works better than simple gray.
                return cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 11, 5)
            
            elif color_hint == 'yellow':
                # Yellow is bright, standard grayscale + CLAHE works best
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                return clahe.apply(gray)
        
        # Default cleanup for other colors or no hint
        # Mix of V channel and CLAHE for best results
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        v = hsv[:,:,2]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(v)
        return enhanced

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
