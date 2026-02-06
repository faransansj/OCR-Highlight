"""
Underline and Strikethrough Detection Module
Uses Hough Line Transform and text proximity analysis
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class LineMarkup:
    """Detected line-based markup (underline or strikethrough)"""
    type: str  # 'underline' or 'strikethrough'
    subtype: str  # 'single', 'double', 'wavy', 'dotted'
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    line_coords: List[Tuple[int, int, int, int]]  # (x1, y1, x2, y2) for each line
    associated_text_bbox: Optional[Tuple[int, int, int, int]] = None


class LineMarkupDetector:
    """Detect underlines and strikethroughs using Hough Line Transform"""
    
    def __init__(self, 
                 angle_tolerance: float = 5.0,
                 min_line_length: int = 20,
                 max_line_gap: int = 10,
                 line_thickness_range: Tuple[int, int] = (1, 5)):
        """
        Args:
            angle_tolerance: Maximum deviation from horizontal (degrees)
            min_line_length: Minimum line length in pixels
            max_line_gap: Maximum gap between line segments
            line_thickness_range: (min, max) thickness in pixels
        """
        self.angle_tolerance = angle_tolerance
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.line_thickness_range = line_thickness_range
        
    def detect(self, image: np.ndarray, 
               text_bboxes: Optional[List[Tuple[int, int, int, int]]] = None) -> List[LineMarkup]:
        """
        Detect all line-based markups in the image
        
        Args:
            image: Input image (BGR or grayscale)
            text_bboxes: Optional list of text bounding boxes for proximity analysis
            
        Returns:
            List of detected LineMarkup objects
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        if lines is None:
            return []
        
        # Filter horizontal lines
        horizontal_lines = self._filter_horizontal_lines(lines)
        
        # Classify as underline or strikethrough based on text proximity
        markups = self._classify_lines(horizontal_lines, text_bboxes, gray.shape)
        
        # Detect double lines
        markups = self._detect_double_lines(markups)
        
        return markups
    
    def _filter_horizontal_lines(self, lines: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Filter lines that are approximately horizontal"""
        horizontal = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle
            if x2 - x1 == 0:
                continue  # Vertical line, skip
            
            angle = np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            
            # Check if approximately horizontal
            if angle <= self.angle_tolerance or angle >= (180 - self.angle_tolerance):
                horizontal.append((x1, y1, x2, y2))
        
        return horizontal
    
    def _classify_lines(self, 
                       lines: List[Tuple[int, int, int, int]], 
                       text_bboxes: Optional[List[Tuple[int, int, int, int]]],
                       image_shape: Tuple[int, int]) -> List[LineMarkup]:
        """Classify lines as underline or strikethrough based on text proximity"""
        markups = []
        
        if text_bboxes is None or len(text_bboxes) == 0:
            # No text info, classify based on vertical position
            # Lines in top half = strikethrough, bottom half = underline (rough heuristic)
            height = image_shape[0]
            
            for x1, y1, x2, y2 in lines:
                avg_y = (y1 + y2) / 2
                line_type = 'strikethrough' if avg_y < height / 2 else 'underline'
                
                markup = LineMarkup(
                    type=line_type,
                    subtype='single',
                    bbox=(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1) + 3),
                    confidence=0.5,  # Low confidence without text info
                    line_coords=[(x1, y1, x2, y2)]
                )
                markups.append(markup)
        else:
            # Analyze proximity to text
            for x1, y1, x2, y2 in lines:
                line_y = (y1 + y2) / 2
                line_x_start = min(x1, x2)
                line_x_end = max(x1, x2)
                
                # Find nearest text bbox
                nearest_bbox, distance, position = self._find_nearest_text(
                    (line_x_start, line_y, line_x_end - line_x_start, 3),
                    text_bboxes
                )
                
                if nearest_bbox is not None:
                    # Classify based on position relative to text
                    if position == 'below':
                        line_type = 'underline'
                        confidence = 0.9 if distance < 10 else 0.7
                    elif position == 'through':
                        line_type = 'strikethrough'
                        confidence = 0.9 if distance < 5 else 0.7
                    else:
                        line_type = 'underline'  # Default
                        confidence = 0.5
                    
                    markup = LineMarkup(
                        type=line_type,
                        subtype='single',
                        bbox=(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1) + 3),
                        confidence=confidence,
                        line_coords=[(x1, y1, x2, y2)],
                        associated_text_bbox=nearest_bbox
                    )
                    markups.append(markup)
        
        return markups
    
    def _find_nearest_text(self, 
                          line_bbox: Tuple[int, int, int, int],
                          text_bboxes: List[Tuple[int, int, int, int]]) -> Tuple[Optional[Tuple], float, str]:
        """
        Find the nearest text bbox and determine relative position
        
        Returns:
            (nearest_bbox, distance, position) where position is 'above', 'below', or 'through'
        """
        lx, ly, lw, lh = line_bbox
        line_center_y = ly + lh / 2
        
        min_distance = float('inf')
        nearest_bbox = None
        position = 'unknown'
        
        for tx, ty, tw, th in text_bboxes:
            # Check horizontal overlap
            if not (lx > tx + tw or lx + lw < tx):
                # Calculate vertical distance
                text_top = ty
                text_bottom = ty + th
                text_center = ty + th / 2
                
                if line_center_y < text_top:
                    # Line is above text
                    distance = text_top - line_center_y
                    pos = 'above'
                elif line_center_y > text_bottom:
                    # Line is below text
                    distance = line_center_y - text_bottom
                    pos = 'below'
                else:
                    # Line goes through text
                    distance = abs(line_center_y - text_center)
                    pos = 'through'
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_bbox = (tx, ty, tw, th)
                    position = pos
        
        return nearest_bbox, min_distance, position
    
    def _detect_double_lines(self, markups: List[LineMarkup]) -> List[LineMarkup]:
        """Detect double underlines/strikethroughs by grouping parallel lines"""
        # Group lines by type and proximity
        grouped = {}
        
        for i, markup in enumerate(markups):
            key = markup.type
            if key not in grouped:
                grouped[key] = []
            grouped[key].append((i, markup))
        
        # Check for parallel pairs
        result = []
        used = set()
        
        for line_type, items in grouped.items():
            for i, (idx1, m1) in enumerate(items):
                if idx1 in used:
                    continue
                
                # Look for parallel line nearby
                found_pair = False
                for j, (idx2, m2) in enumerate(items[i+1:], start=i+1):
                    if idx2 in used:
                        continue
                    
                    # Check if parallel and close
                    y1 = m1.line_coords[0][1]
                    y2 = m2.line_coords[0][1]
                    
                    if abs(y1 - y2) < 10:  # Within 10 pixels vertically
                        # Merge into double line
                        merged = LineMarkup(
                            type=m1.type,
                            subtype='double',
                            bbox=(
                                min(m1.bbox[0], m2.bbox[0]),
                                min(m1.bbox[1], m2.bbox[1]),
                                max(m1.bbox[0] + m1.bbox[2], m2.bbox[0] + m2.bbox[2]) - min(m1.bbox[0], m2.bbox[0]),
                                max(m1.bbox[1] + m1.bbox[3], m2.bbox[1] + m2.bbox[3]) - min(m1.bbox[1], m2.bbox[1])
                            ),
                            confidence=min(m1.confidence, m2.confidence),
                            line_coords=m1.line_coords + m2.line_coords,
                            associated_text_bbox=m1.associated_text_bbox
                        )
                        result.append(merged)
                        used.add(idx1)
                        used.add(idx2)
                        found_pair = True
                        break
                
                if not found_pair:
                    result.append(m1)
                    used.add(idx1)
        
        return result


if __name__ == "__main__":
    # Test code
    import sys
    
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        detector = LineMarkupDetector()
        markups = detector.detect(img)
        
        print(f"Detected {len(markups)} line markups:")
        for m in markups:
            print(f"  - {m.type} ({m.subtype}): bbox={m.bbox}, conf={m.confidence:.2f}")
        
        # Visualize
        vis = img.copy()
        for m in markups:
            color = (0, 255, 0) if m.type == 'underline' else (0, 0, 255)
            x, y, w, h = m.bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            cv2.putText(vis, f"{m.type} ({m.subtype})", (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        cv2.imwrite('line_markup_detection.png', vis)
        print("\nVisualization saved to line_markup_detection.png")
