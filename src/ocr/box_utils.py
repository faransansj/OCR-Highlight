"""
Bounding Box Utilities
Tools for splitting and refining markup regions
"""

import cv2
import numpy as np
from typing import List

class BoxSplitter:
    """Logic to split a large bounding box into smaller word-level components"""

    @staticmethod
    def split_by_vertical_projection(image: np.ndarray, 
                                   padding: int = 2) -> List[List[int]]:
        """
        Split a region into sub-boxes based on vertical gaps
        """
        if image is None or image.size == 0:
            return []

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Pre-process to merge characters into words
        # Binary threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological closing to join characters in a word
        # 7px width kernel should bridge most character gaps but not word gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Vertical projection
        projection = np.sum(closed, axis=0)
        
        # Identify gaps
        mask = projection > 0
        
        # Find continuous segments
        sub_boxes = []
        in_segment = False
        start_x = 0
        h, w = binary.shape
        
        for x in range(w):
            if not in_segment and mask[x]:
                start_x = x
                in_segment = True
            elif in_segment and not mask[x]:
                end_x = x
                
                # Minimum word width
                if end_x - start_x < 5: continue

                # Use original binary to find precise Y range
                segment_binary = binary[:, start_x:end_x]
                y_projection = np.sum(segment_binary, axis=1)
                y_mask = y_projection > 0
                
                if np.any(y_mask):
                    y_indices = np.where(y_mask)[0]
                    start_y = y_indices[0]
                    end_y = y_indices[-1]
                    
                    sx = max(0, start_x - padding)
                    sy = max(0, start_y - padding)
                    sw = min(w, end_x + padding) - sx
                    sh = min(h, end_y + padding) - sy
                    
                    sub_boxes.append([sx, sy, sw, sh])
                
                in_segment = False
        
        # Handle last segment
        if in_segment:
            end_x = w
            segment_binary = binary[:, start_x:end_x]
            y_projection = np.sum(segment_binary, axis=1)
            y_mask = y_projection > 0
            if np.any(y_mask):
                y_indices = np.where(y_mask)[0]
                sub_boxes.append([
                    max(0, start_x - padding),
                    max(0, y_indices[0] - padding),
                    min(w, end_x + padding) - max(0, start_x - padding),
                    min(h, y_indices[-1] + padding) - max(0, y_indices[0] - padding)
                ])

        return sub_boxes
