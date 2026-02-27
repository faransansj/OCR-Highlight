"""
Automatic Markup Generator for Real Document Images
Adds synthetic markups (highlights, underlines, circles, rectangles) to real documents
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Dict
import random
from dataclasses import dataclass, asdict


@dataclass
class MarkupAnnotation:
    """Annotation for a single markup"""
    markup_type: str
    bbox: List[int]  # [x, y, width, height]
    confidence: float = 1.0
    properties: Dict = None


class AutoMarkupGenerator:
    """
    Automatically adds realistic markups to real document images
    """
    
    def __init__(self, 
                 highlight_alpha: float = 0.3,
                 line_thickness: int = 3,
                 shape_thickness: int = 3):
        """
        Args:
            highlight_alpha: Transparency for highlights (0-1)
            line_thickness: Thickness of underlines/strikethroughs
            shape_thickness: Thickness of shapes (circles, rectangles)
        """
        self.highlight_alpha = highlight_alpha
        self.line_thickness = line_thickness
        self.shape_thickness = shape_thickness
        
        # Markup colors
        self.colors = {
            'highlight_yellow': (0, 255, 255),
            'highlight_green': (0, 255, 0),
            'highlight_pink': (180, 105, 255),
            'underline_red': (0, 0, 255),
            'underline_blue': (255, 0, 0),
            'circle_red': (0, 0, 255),
            'circle_green': (0, 255, 0),
            'rectangle_blue': (255, 0, 0),
            'rectangle_green': (0, 255, 0),
        }
    
    def add_highlight(self, img: np.ndarray, x: int, y: int, w: int, h: int, 
                     color_name: str = 'highlight_yellow') -> MarkupAnnotation:
        """Add semi-transparent highlight"""
        overlay = img.copy()
        color = self.colors.get(color_name, self.colors['highlight_yellow'])
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        cv2.addWeighted(overlay, self.highlight_alpha, img, 1 - self.highlight_alpha, 0, img)
        
        return MarkupAnnotation(
            markup_type='highlight',
            bbox=[x, y, w, h],
            properties={'color': color_name}
        )
    
    def add_underline(self, img: np.ndarray, x: int, y: int, w: int, h: int,
                     double: bool = False, color_name: str = 'underline_red') -> MarkupAnnotation:
        """Add underline below text region"""
        color = self.colors.get(color_name, self.colors['underline_red'])
        y_line = y + h + 5
        
        cv2.line(img, (x, y_line), (x + w, y_line), color, self.line_thickness)
        
        if double:
            cv2.line(img, (x, y_line + 6), (x + w, y_line + 6), color, self.line_thickness)
        
        return MarkupAnnotation(
            markup_type='underline' if not double else 'double_underline',
            bbox=[x, y, w, h + 15],
            properties={'color': color_name, 'double': double}
        )
    
    def add_strikethrough(self, img: np.ndarray, x: int, y: int, w: int, h: int,
                         color_name: str = 'underline_red') -> MarkupAnnotation:
        """Add strikethrough through text region"""
        color = self.colors.get(color_name, self.colors['underline_red'])
        y_mid = y + h // 2
        
        cv2.line(img, (x, y_mid), (x + w, y_mid), color, self.line_thickness)
        
        return MarkupAnnotation(
            markup_type='strikethrough',
            bbox=[x, y, w, h],
            properties={'color': color_name}
        )
    
    def add_circle(self, img: np.ndarray, x: int, y: int, w: int, h: int,
                  color_name: str = 'circle_red') -> MarkupAnnotation:
        """Add circle around region"""
        center_x = x + w // 2
        center_y = y + h // 2
        radius = max(w, h) // 2 + 15
        color = self.colors.get(color_name, self.colors['circle_red'])
        
        cv2.circle(img, (center_x, center_y), radius, color, self.shape_thickness)
        
        return MarkupAnnotation(
            markup_type='circle',
            bbox=[center_x - radius, center_y - radius, radius * 2, radius * 2],
            properties={'color': color_name, 'center': [center_x, center_y], 'radius': radius}
        )
    
    def add_rectangle(self, img: np.ndarray, x: int, y: int, w: int, h: int,
                     color_name: str = 'rectangle_blue') -> MarkupAnnotation:
        """Add rectangle around region"""
        padding = 15
        color = self.colors.get(color_name, self.colors['rectangle_blue'])
        
        cv2.rectangle(img, (x - padding, y - padding), 
                     (x + w + padding, y + h + padding), color, self.shape_thickness)
        
        return MarkupAnnotation(
            markup_type='rectangle',
            bbox=[x - padding, y - padding, w + 2 * padding, h + 2 * padding],
            properties={'color': color_name}
        )
    
    def generate_random_regions(self, img_height: int, img_width: int, 
                               num_regions: int = 3) -> List[Tuple[int, int, int, int]]:
        """
        Generate random regions that look like text areas
        """
        regions = []
        
        for _ in range(num_regions):
            # Random region size (typical text line dimensions)
            w = random.randint(100, min(400, img_width - 100))
            h = random.randint(20, 60)
            
            # Random position (avoid edges)
            x = random.randint(50, img_width - w - 50)
            y = random.randint(50, img_height - h - 50)
            
            regions.append((x, y, w, h))
        
        return regions
    
    def process_image(self, img_path: str, output_path: str, 
                     num_markups: int = 3, markup_types: List[str] = None) -> Dict:
        """
        Process a single image: add markups and save annotated version + GT
        
        Args:
            img_path: Input image path
            output_path: Output image path
            num_markups: Number of markups to add
            markup_types: Types to use ['highlight', 'underline', 'circle', 'rectangle']
        
        Returns:
            Dictionary with image metadata and annotations
        """
        if markup_types is None:
            markup_types = ['highlight', 'underline', 'circle', 'rectangle', 'strikethrough']
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        h, w = img.shape[:2]
        
        # Generate random regions
        regions = self.generate_random_regions(h, w, num_markups)
        
        # Add markups
        annotations = []
        for region in regions:
            x, y, width, height = region
            
            # Random markup type
            markup_type = random.choice(markup_types)
            
            if markup_type == 'highlight':
                color_choice = random.choice(['highlight_yellow', 'highlight_green', 'highlight_pink'])
                ann = self.add_highlight(img, x, y, width, height, color_choice)
            elif markup_type == 'underline':
                double = random.random() < 0.3
                color_choice = random.choice(['underline_red', 'underline_blue'])
                ann = self.add_underline(img, x, y, width, height, double, color_choice)
            elif markup_type == 'strikethrough':
                color_choice = random.choice(['underline_red', 'underline_blue'])
                ann = self.add_strikethrough(img, x, y, width, height, color_choice)
            elif markup_type == 'circle':
                color_choice = random.choice(['circle_red', 'circle_green'])
                ann = self.add_circle(img, x, y, width, height, color_choice)
            elif markup_type == 'rectangle':
                color_choice = random.choice(['rectangle_blue', 'rectangle_green'])
                ann = self.add_rectangle(img, x, y, width, height, color_choice)
            
            annotations.append(ann)
        
        # Save annotated image
        cv2.imwrite(output_path, img)
        
        # Prepare metadata
        metadata = {
            'source_image': str(img_path),
            'output_image': str(output_path),
            'image_size': {'width': w, 'height': h},
            'num_markups': len(annotations),
            'annotations': [asdict(ann) for ann in annotations]
        }
        
        return metadata
    
    def process_batch(self, input_dir: str, output_dir: str, 
                     num_images: int = 100, markups_per_image: int = 3) -> List[Dict]:
        """
        Process a batch of images
        
        Args:
            input_dir: Directory containing source images
            output_dir: Directory for annotated images + GT JSON
            num_images: Number of images to process
            markups_per_image: Number of markups per image
        
        Returns:
            List of metadata dictionaries
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get image files
        image_files = list(input_path.glob('*.png')) + list(input_path.glob('*.jpg'))
        image_files = image_files[:num_images]
        
        all_metadata = []
        
        for i, img_file in enumerate(image_files):
            output_img = output_path / f'marked_{img_file.stem}.png'
            output_json = output_path / f'marked_{img_file.stem}.json'
            
            try:
                metadata = self.process_image(
                    str(img_file), 
                    str(output_img), 
                    num_markups=markups_per_image
                )
                
                # Save JSON
                with open(output_json, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                all_metadata.append(metadata)
                
                if (i + 1) % 10 == 0:
                    print(f'Processed {i+1}/{len(image_files)} images...')
            
            except Exception as e:
                print(f'Error processing {img_file}: {e}')
        
        # Save manifest
        manifest_path = output_path / 'manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump({
                'total_images': len(all_metadata),
                'markups_per_image': markups_per_image,
                'images': all_metadata
            }, f, indent=2)
        
        print(f'✅ Batch processing complete: {len(all_metadata)} images')
        print(f'   Output: {output_path}/')
        print(f'   Manifest: {manifest_path}')
        
        return all_metadata


if __name__ == '__main__':
    # Example usage
    generator = AutoMarkupGenerator()
    
    # Process batch
    metadata = generator.process_batch(
        input_dir='data/docvqa_samples',
        output_dir='data/docvqa_with_markups',
        num_images=100,
        markups_per_image=3
    )
    
    print(f'\n✅ Generated {len(metadata)} annotated images!')
