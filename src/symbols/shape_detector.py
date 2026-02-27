"""
Shape-based Markup Detection Module
Detects circles, rectangles, stars, checkmarks using contour analysis
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class ShapeMarkup:
    """Detected shape-based markup"""
    shape_type: str  # 'circle', 'rectangle', 'star', 'checkmark'
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    contour: np.ndarray
    center: Tuple[int, int]
    properties: dict  # Additional shape-specific properties


class ShapeDetector:
    """Detect shape-based markups (circles, boxes, stars, checks)"""
    
    def __init__(self,
                 min_area: int = 500,  # Increased from 100 to reduce FP
                 max_area: int = 10000,
                 circularity_threshold: float = 0.85,
                 rectangularity_threshold: float = 0.85,
                 min_confidence: float = 0.65):  # New: minimum confidence threshold
        """
        Args:
            min_area: Minimum contour area in pixels
            max_area: Maximum contour area in pixels
            circularity_threshold: Minimum circularity for circle detection
            rectangularity_threshold: Minimum rectangularity for box detection
        """
        self.min_area = min_area
        self.max_area = max_area
        self.circularity_threshold = circularity_threshold
        self.rectangularity_threshold = rectangularity_threshold
        self.min_confidence = min_confidence
        
    def detect(self, image: np.ndarray) -> List[ShapeMarkup]:
        """
        Detect all shape markups in the image
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            List of detected ShapeMarkup objects
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        markups = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate center
            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            # Try to classify shape
            shape_type, confidence, properties = self._classify_shape(contour, area)
            
            if shape_type is not None and confidence >= self.min_confidence:
                markup = ShapeMarkup(
                    shape_type=shape_type,
                    bbox=(x, y, w, h),
                    confidence=confidence,
                    contour=contour,
                    center=(cx, cy),
                    properties=properties
                )
                markups.append(markup)
        
        return markups
    
    def _classify_shape(self, contour: np.ndarray, area: float) -> Tuple[Optional[str], float, dict]:
        """
        Classify the shape type
        
        Returns:
            (shape_type, confidence, properties)
        """
        # Calculate perimeter
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return None, 0.0, {}
        
        # Calculate circularity: 4π * area / perimeter^2
        # Perfect circle = 1.0
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        
        # Get approximate polygon
        epsilon = 0.04 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        num_vertices = len(approx)
        
        # Get bounding box aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h != 0 else 0
        
        # Calculate extent (contour area / bounding box area)
        bbox_area = w * h
        extent = float(area) / bbox_area if bbox_area != 0 else 0
        
        # === Circle Detection ===
        if circularity >= self.circularity_threshold:
            return 'circle', circularity, {
                'circularity': circularity,
                'radius': math.sqrt(area / math.pi)
            }
        
        # === Rectangle Detection ===
        if num_vertices == 4 and extent > self.rectangularity_threshold:
            # Check if angles are approximately 90 degrees
            angles = self._get_polygon_angles(approx)
            if all(80 <= angle <= 100 for angle in angles):
                conf = min(extent, 1.0)
                return 'rectangle', conf, {
                    'aspect_ratio': aspect_ratio,
                    'extent': extent,
                    'angles': angles
                }
        
        # === Star Detection ===
        # Stars typically have 5-6 vertices with alternating long/short distances
        if 5 <= num_vertices <= 6:
            is_star, star_conf = self._check_star_pattern(approx)
            if is_star:
                return 'star', star_conf, {
                    'num_points': num_vertices,
                    'circularity': circularity
                }
        
        # === Checkmark Detection ===
        # Checkmarks are typically open contours with 2-3 segments
        if 2 <= num_vertices <= 4 and extent < 0.6:
            is_check, check_conf = self._check_checkmark_pattern(approx)
            if is_check:
                return 'checkmark', check_conf, {
                    'num_vertices': num_vertices,
                    'extent': extent
                }
        
        # Unknown shape
        return None, 0.0, {}
    
    def _get_polygon_angles(self, approx: np.ndarray) -> List[float]:
        """Calculate interior angles of a polygon"""
        angles = []
        n = len(approx)
        
        for i in range(n):
            p1 = approx[i][0]
            p2 = approx[(i + 1) % n][0]
            p3 = approx[(i + 2) % n][0]
            
            # Vectors
            v1 = p1 - p2
            v2 = p3 - p2
            
            # Angle between vectors
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            
            angles.append(angle)
        
        return angles
    
    def _check_star_pattern(self, approx: np.ndarray) -> Tuple[bool, float]:
        """
        Check if polygon has star-like pattern (alternating long/short distances from center)
        
        Returns:
            (is_star, confidence)
        """
        # Calculate centroid
        M = cv2.moments(approx)
        if M['m00'] == 0:
            return False, 0.0
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
        center = np.array([cx, cy])
        
        # Calculate distances from center to each vertex
        distances = []
        for point in approx:
            p = point[0]
            dist = np.linalg.norm(p - center)
            distances.append(dist)
        
        if len(distances) < 5:
            return False, 0.0
        
        # Check for alternating pattern
        sorted_dist = sorted(distances)
        median = sorted_dist[len(sorted_dist) // 2]
        
        # Count alternating long/short
        alternations = 0
        for i in range(len(distances) - 1):
            if (distances[i] > median) != (distances[i + 1] > median):
                alternations += 1
        
        # Star should have high alternation rate
        alternation_rate = alternations / len(distances)
        confidence = alternation_rate if alternation_rate > 0.6 else 0.0
        
        return alternation_rate > 0.6, confidence
    
    def _check_checkmark_pattern(self, approx: np.ndarray) -> Tuple[bool, float]:
        """
        Check if polygon has checkmark-like pattern (V or tick shape)
        
        Returns:
            (is_checkmark, confidence)
        """
        if len(approx) < 2 or len(approx) > 4:
            return False, 0.0
        
        # Checkmarks typically have:
        # 1. A sharp corner (< 90 degrees)
        # 2. Two segments with different lengths
        
        angles = self._get_polygon_angles(approx)
        
        # Find sharpest angle
        min_angle = min(angles) if angles else 180
        
        # Checkmark should have at least one sharp angle
        if min_angle < 60:
            confidence = 1.0 - (min_angle / 60)  # Sharper = higher confidence
            return True, confidence
        
        return False, 0.0


if __name__ == "__main__":
    # Test code
    import sys
    
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        detector = ShapeDetector()
        shapes = detector.detect(img)
        
        print(f"Detected {len(shapes)} shapes:")
        for s in shapes:
            print(f"  - {s.shape_type}: bbox={s.bbox}, conf={s.confidence:.2f}, props={s.properties}")
        
        # Visualize
        vis = img.copy()
        colors = {
            'circle': (255, 0, 0),      # Blue
            'rectangle': (0, 255, 0),   # Green
            'star': (0, 255, 255),      # Yellow
            'checkmark': (255, 0, 255)  # Magenta
        }
        
        for s in shapes:
            color = colors.get(s.shape_type, (128, 128, 128))
            x, y, w, h = s.bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            cv2.putText(vis, f"{s.shape_type} ({s.confidence:.2f})", (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw contour
            cv2.drawContours(vis, [s.contour], -1, color, 2)
        
        cv2.imwrite('shape_detection.png', vis)
        print("\nVisualization saved to shape_detection.png")
