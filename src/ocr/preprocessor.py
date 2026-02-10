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

        # 1. Convert to HLS to better separate color from luminosity
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        h, l, s = cv2.split(hls)
        
        # 2. Extract Luminosity (L channel) - often has best contrast for text
        # But for bright highlights, the background might be too bright
        
        # 3. Targeted color removal (The "Purify" spell)
        # We want to boost contrast between black text and colored background
        if color_hint:
            # Create a mask for the highlight color
            # Typical HSV/HLS ranges for highlights:
            # Yellow: H=30, Pink: H=150-170, Green: H=60, Blue: H=100-120
            
            # Simple approach: use the L channel but enhance it
            # High-pass filter to remove slow background variations
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Denoising
            denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
            
            # For very specific colors, we can do background subtraction in HLS space
            if color_hint == 'pink' or color_hint == 'red':
                # Pink is tough, often overlaps with black text in simple grayscale
                # Use L channel and threshold aggressively
                _, thresh = cv2.threshold(l, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                return thresh
                
            elif color_hint == 'yellow':
                # Yellow is bright, simple grayscale + contrast boost works well
                return enhanced
                
            return denoised
        else:
            # Standard cleanup
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            denoised = cv2.GaussianBlur(gray, (3, 3), 0)
            _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return thresh

    @staticmethod
    def remove_lines(image: np.ndarray) -> np.ndarray:
        """
        Spell to remove underlines or strikethroughs while keeping text
        Useful when OCR gets confused by horizontal lines
        """
        # Morphological operations to detect and remove horizontal lines
        # This is high-level magic!
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
        # We use bitwise XOR to remove, then invert back
        cleaned_inv = cv2.bitwise_and(binary, cv2.bitwise_not(detected_lines))
        cleaned = cv2.bitwise_not(cleaned_inv)
        
        return cleaned
