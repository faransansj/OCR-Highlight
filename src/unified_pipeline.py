"""
Unified Markup Detection Pipeline
Integrates all markup detectors (highlights, lines, shapes) into a single pipeline
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import json
from pathlib import Path

# Import existing detectors
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from highlight_detector import HighlightDetector  # Original
from symbols.line_detector import LineMarkupDetector, LineMarkup
from symbols.shape_detector import ShapeDetector, ShapeMarkup


@dataclass
class UnifiedMarkup:
    """Unified markup detection result"""
    markup_type: str  # 'highlight', 'underline', 'strikethrough', 'circle', etc.
    subtype: str  # color name, 'single', 'double', etc.
    bbox: Tuple[int, int, int, int]
    confidence: float
    text: Optional[str] = None
    properties: Optional[Dict] = None


class UnifiedMarkupPipeline:
    """
    End-to-end pipeline for detecting all markup types
    Priority: highlights > shapes > underlines > strikethroughs
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize all detectors
        
        Args:
            config_path: Path to configuration JSON (optional)
        """
        # Load config if provided
        self.config = self._load_config(config_path) if config_path else {}
        
        # Initialize detectors
        self.highlight_detector = HighlightDetector()
        self.line_detector = LineMarkupDetector()
        self.shape_detector = ShapeDetector()
        
        # OCR engine (optional, for text extraction)
        self.ocr_engine = None
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def process_image(self, image_path: str) -> Dict:
        """
        Process an image and detect all markups
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary with detection results
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Detect all markups
        all_markups = []
        
        # 1. Detect highlights (highest priority)
        highlights = self._detect_highlights(img)
        all_markups.extend(highlights)
        
        # 2. Detect shapes
        shapes = self._detect_shapes(img)
        all_markups.extend(shapes)
        
        # 3. Detect lines (underlines, strikethroughs)
        lines = self._detect_lines(img)
        all_markups.extend(lines)
        
        # Extract text from marked regions (if OCR enabled)
        if self.ocr_engine:
            all_markups = self._extract_text(img, all_markups)
        
        # Remove overlaps (priority-based)
        all_markups = self._resolve_overlaps(all_markups)
        
        # Build result
        result = {
            'image_path': image_path,
            'total_markups': len(all_markups),
            'markups_by_type': self._group_by_type(all_markups),
            'markups': [asdict(m) for m in all_markups]
        }
        
        return result
    
    def _detect_highlights(self, img: np.ndarray) -> List[UnifiedMarkup]:
        """Detect color highlights"""
        # Use existing highlight detector
        highlights_raw = self.highlight_detector.detect_highlights(img)
        
        markups = []
        for h in highlights_raw:
            markup = UnifiedMarkup(
                markup_type='highlight',
                subtype=h.get('color', 'unknown'),
                bbox=h['bbox'],
                confidence=h.get('confidence', 1.0),
                text=h.get('text'),
                properties={'color': h.get('color')}
            )
            markups.append(markup)
        
        return markups
    
    def _detect_shapes(self, img: np.ndarray) -> List[UnifiedMarkup]:
        """Detect shape markups"""
        shapes: List[ShapeMarkup] = self.shape_detector.detect(img)
        
        markups = []
        for s in shapes:
            markup = UnifiedMarkup(
                markup_type=s.shape_type,
                subtype='standard',
                bbox=s.bbox,
                confidence=s.confidence,
                properties=s.properties
            )
            markups.append(markup)
        
        return markups
    
    def _detect_lines(self, img: np.ndarray) -> List[UnifiedMarkup]:
        """Detect line markups (underlines, strikethroughs)"""
        lines: List[LineMarkup] = self.line_detector.detect(img)
        
        markups = []
        for line in lines:
            markup = UnifiedMarkup(
                markup_type=line.type,
                subtype=line.subtype,
                bbox=line.bbox,
                confidence=line.confidence,
                properties={
                    'line_coords': line.line_coords,
                    'associated_text_bbox': line.associated_text_bbox
                }
            )
            markups.append(markup)
        
        return markups
    
    def _extract_text(self, img: np.ndarray, markups: List[UnifiedMarkup]) -> List[UnifiedMarkup]:
        """Extract text from marked regions using OCR"""
        # TODO: Integrate OCR engine
        # For now, return markups unchanged
        return markups
    
    def _resolve_overlaps(self, markups: List[UnifiedMarkup]) -> List[UnifiedMarkup]:
        """
        Resolve overlapping markups based on priority
        Priority: highlight > circle > rectangle > underline > strikethrough
        """
        priority_map = {
            'highlight': 1,
            'circle': 2,
            'rectangle': 3,
            'star': 4,
            'checkmark': 5,
            'underline': 6,
            'strikethrough': 7
        }
        
        # Sort by priority
        markups_sorted = sorted(markups, key=lambda m: priority_map.get(m.markup_type, 99))
        
        # Remove overlaps
        kept = []
        for m in markups_sorted:
            overlaps = False
            for existing in kept:
                if self._boxes_overlap(m.bbox, existing.bbox):
                    overlaps = True
                    break
            
            if not overlaps:
                kept.append(m)
        
        return kept
    
    def _boxes_overlap(self, box1: Tuple[int, int, int, int], 
                      box2: Tuple[int, int, int, int], 
                      threshold: float = 0.5) -> bool:
        """Check if two bounding boxes overlap significantly"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return False
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = w1 * h1
        box2_area = w2 * h2
        
        # IoU (Intersection over Union)
        iou = intersection_area / (box1_area + box2_area - intersection_area)
        
        return iou > threshold
    
    def _group_by_type(self, markups: List[UnifiedMarkup]) -> Dict[str, int]:
        """Group markups by type and count"""
        counts = {}
        for m in markups:
            counts[m.markup_type] = counts.get(m.markup_type, 0) + 1
        return counts
    
    def save_results(self, result: Dict, output_path: str):
        """Save detection results to JSON"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    
    def visualize(self, image_path: str, result: Dict, output_path: str):
        """Visualize detection results"""
        img = cv2.imread(image_path)
        
        # Color map for different markup types
        colors = {
            'highlight': (0, 255, 255),     # Yellow
            'underline': (0, 255, 0),       # Green
            'strikethrough': (0, 0, 255),   # Red
            'circle': (255, 0, 0),          # Blue
            'rectangle': (255, 255, 0),     # Cyan
            'star': (255, 0, 255),          # Magenta
            'checkmark': (128, 0, 128)      # Purple
        }
        
        for m in result['markups']:
            markup_type = m['markup_type']
            x, y, w, h = m['bbox']
            color = colors.get(markup_type, (128, 128, 128))
            
            # Draw box
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{markup_type} ({m['subtype']})"
            cv2.putText(img, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        cv2.imwrite(output_path, img)
        print(f"Visualization saved to {output_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Process a single image
        pipeline = UnifiedMarkupPipeline()
        result = pipeline.process_image(sys.argv[1])
        
        print(json.dumps(result, indent=2))
        
        # Visualize
        output_vis = sys.argv[1].replace('.', '_detected.')
        pipeline.visualize(sys.argv[1], result, output_vis)
    else:
        print("Usage: python unified_pipeline.py <image_path>")
