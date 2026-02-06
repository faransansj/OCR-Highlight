"""
Advanced Synthetic Data Generation Pipeline
Generates documents with multiple markup types (highlights, underlines, shapes, etc.)
Supports multiple languages and document styles
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import string


@dataclass
class MarkupAnnotation:
    """Ground truth annotation for a markup"""
    markup_type: str  # 'highlight', 'underline', 'strikethrough', 'circle', etc.
    subtype: str  # 'yellow', 'single', 'double', etc.
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    text: Optional[str] = None  # Associated text (if applicable)
    color: Optional[Tuple[int, int, int]] = None  # RGB color
    

@dataclass
class DocumentSample:
    """A generated document sample with ground truth"""
    image_path: str
    annotations: List[MarkupAnnotation]
    metadata: Dict


class SyntheticDataGenerator:
    """Generate synthetic document images with various markups"""
    
    def __init__(self, 
                 output_dir: str = "data/synthetic_v2",
                 languages: List[str] = None,
                 fonts_dir: str = None):
        """
        Args:
            output_dir: Directory to save generated data
            languages: List of language codes (default: ['eng', 'kor'])
            fonts_dir: Directory containing font files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.languages = languages or ['eng', 'kor', 'jpn', 'chi_sim']
        self.fonts_dir = Path(fonts_dir) if fonts_dir else None
        
        # Markup color palettes
        self.highlight_colors = {
            'yellow': (255, 255, 0),
            'green': (144, 238, 144),
            'pink': (255, 182, 193),
            'blue': (173, 216, 230),
            'orange': (255, 200, 124)
        }
        
        # Sample texts by language
        self.sample_texts = self._load_sample_texts()
        
    def _load_sample_texts(self) -> Dict[str, List[str]]:
        """Load sample text snippets for each language"""
        texts = {
            'eng': [
                "Computer vision is a field of artificial intelligence.",
                "Machine learning algorithms can detect patterns in data.",
                "Deep neural networks have revolutionized image processing.",
                "Object detection is a fundamental task in computer vision.",
                "Optical character recognition enables text extraction from images."
            ],
            'kor': [
                "컴퓨터 비전은 인공지능의 한 분야입니다.",
                "머신러닝 알고리즘은 데이터의 패턴을 감지할 수 있습니다.",
                "딥러닝 신경망은 이미지 처리를 혁신했습니다.",
                "객체 감지는 컴퓨터 비전의 기본 작업입니다.",
                "광학 문자 인식은 이미지에서 텍스트 추출을 가능하게 합니다."
            ],
            'jpn': [
                "コンピュータビジョンは人工知能の分野です。",
                "機械学習アルゴリズムはデータのパターンを検出できます。",
                "ディープニューラルネットワークは画像処理を革新しました。"
            ],
            'chi_sim': [
                "计算机视觉是人工智能的一个领域。",
                "机器学习算法可以检测数据中的模式。",
                "深度神经网络彻底改变了图像处理。"
            ]
        }
        return texts
    
    def generate_batch(self, 
                      count: int = 100,
                      markup_types: List[str] = None,
                      max_markups_per_image: int = 5) -> List[DocumentSample]:
        """
        Generate a batch of synthetic documents
        
        Args:
            count: Number of images to generate
            markup_types: Types of markups to include (None = all)
            max_markups_per_image: Maximum markups per image
            
        Returns:
            List of DocumentSample objects
        """
        if markup_types is None:
            markup_types = ['highlight', 'underline', 'strikethrough', 'circle', 'rectangle']
        
        samples = []
        
        for i in range(count):
            # Random parameters
            language = random.choice(self.languages)
            num_lines = random.randint(3, 8)
            num_markups = random.randint(1, max_markups_per_image)
            
            # Generate document
            sample = self._generate_single_document(
                index=i,
                language=language,
                num_lines=num_lines,
                num_markups=num_markups,
                markup_types=markup_types
            )
            
            samples.append(sample)
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{count} samples...")
        
        # Save manifest
        self._save_manifest(samples)
        
        return samples
    
    def _generate_single_document(self,
                                  index: int,
                                  language: str,
                                  num_lines: int,
                                  num_markups: int,
                                  markup_types: List[str]) -> DocumentSample:
        """Generate a single document image with markups"""
        
        # Image parameters
        width, height = 800, 600
        bg_color = (255, 255, 255)
        text_color = (0, 0, 0)
        
        # Create PIL image for text rendering
        pil_img = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(pil_img, 'RGBA')  # RGBA for transparency
        
        # Load font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        # Generate text lines
        texts = random.sample(self.sample_texts.get(language, self.sample_texts['eng']), 
                            min(num_lines, len(self.sample_texts.get(language, []))))
        
        line_bboxes = []
        y_offset = 50
        
        for text in texts:
            x_offset = 50
            bbox = draw.textbbox((x_offset, y_offset), text, font=font)
            draw.text((x_offset, y_offset), text, fill=text_color, font=font)
            line_bboxes.append({
                'text': text,
                'bbox': bbox  # (left, top, right, bottom)
            })
            y_offset += 60
        
        # Convert to numpy for OpenCV operations
        img_np = np.array(pil_img)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Apply markups
        annotations = []
        
        for _ in range(num_markups):
            markup_type = random.choice(markup_types)
            
            # Select a random line to markup
            if not line_bboxes:
                continue
            
            target_line = random.choice(line_bboxes)
            left, top, right, bottom = target_line['bbox']
            text_width = right - left
            text_height = bottom - top
            
            # Apply markup based on type
            if markup_type == 'highlight':
                annotation = self._add_highlight(img_np, left, top, text_width, text_height, target_line['text'])
            elif markup_type == 'underline':
                annotation = self._add_underline(img_np, left, top, text_width, text_height, target_line['text'])
            elif markup_type == 'strikethrough':
                annotation = self._add_strikethrough(img_np, left, top, text_width, text_height, target_line['text'])
            elif markup_type == 'circle':
                annotation = self._add_circle(img_np, left, top, text_width, text_height, target_line['text'])
            elif markup_type == 'rectangle':
                annotation = self._add_rectangle(img_np, left, top, text_width, text_height, target_line['text'])
            else:
                continue
            
            if annotation:
                annotations.append(annotation)
        
        # Add noise/variations
        img_np = self._add_variations(img_np)
        
        # Save image
        filename = f"sample_{index:06d}_{language}.png"
        filepath = self.output_dir / filename
        cv2.imwrite(str(filepath), img_np)
        
        # Create sample object
        sample = DocumentSample(
            image_path=str(filepath),
            annotations=annotations,
            metadata={
                'language': language,
                'num_lines': num_lines,
                'num_markups': len(annotations),
                'index': index
            }
        )
        
        # Save annotation JSON
        json_path = self.output_dir / f"sample_{index:06d}_{language}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'image_path': str(filepath),
                'annotations': [asdict(a) for a in annotations],
                'metadata': sample.metadata
            }, f, indent=2, ensure_ascii=False)
        
        return sample
    
    def _add_highlight(self, img: np.ndarray, x: int, y: int, w: int, h: int, text: str) -> MarkupAnnotation:
        """Add highlight markup"""
        color_name = random.choice(list(self.highlight_colors.keys()))
        color_rgb = self.highlight_colors[color_name]
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        
        # Create semi-transparent overlay
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color_bgr, -1)
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        return MarkupAnnotation(
            markup_type='highlight',
            subtype=color_name,
            bbox=(x, y, w, h),
            text=text,
            color=color_rgb
        )
    
    def _add_underline(self, img: np.ndarray, x: int, y: int, w: int, h: int, text: str) -> MarkupAnnotation:
        """Add underline markup"""
        line_type = random.choice(['single', 'double'])
        color = (0, 0, 0)
        thickness = 2
        
        y_line = y + h + 5
        
        if line_type == 'single':
            cv2.line(img, (x, y_line), (x + w, y_line), color, thickness)
        else:  # double
            cv2.line(img, (x, y_line), (x + w, y_line), color, thickness)
            cv2.line(img, (x, y_line + 4), (x + w, y_line + 4), color, thickness)
        
        return MarkupAnnotation(
            markup_type='underline',
            subtype=line_type,
            bbox=(x, y_line - 2, w, 10 if line_type == 'double' else 4),
            text=text
        )
    
    def _add_strikethrough(self, img: np.ndarray, x: int, y: int, w: int, h: int, text: str) -> MarkupAnnotation:
        """Add strikethrough markup"""
        line_type = random.choice(['single', 'double'])
        color = (0, 0, 255)  # Red
        thickness = 2
        
        y_line = y + h // 2
        
        if line_type == 'single':
            cv2.line(img, (x, y_line), (x + w, y_line), color, thickness)
        else:  # double
            cv2.line(img, (x, y_line - 2), (x + w, y_line - 2), color, thickness)
            cv2.line(img, (x, y_line + 2), (x + w, y_line + 2), color, thickness)
        
        return MarkupAnnotation(
            markup_type='strikethrough',
            subtype=line_type,
            bbox=(x, y_line - 4, w, 8),
            text=text
        )
    
    def _add_circle(self, img: np.ndarray, x: int, y: int, w: int, h: int, text: str) -> MarkupAnnotation:
        """Add circle markup around text"""
        center_x = x + w // 2
        center_y = y + h // 2
        radius = max(w, h) // 2 + 10
        color = (255, 0, 0)  # Blue
        thickness = 2
        
        cv2.circle(img, (center_x, center_y), radius, color, thickness)
        
        return MarkupAnnotation(
            markup_type='circle',
            subtype='standard',
            bbox=(center_x - radius, center_y - radius, radius * 2, radius * 2),
            text=text
        )
    
    def _add_rectangle(self, img: np.ndarray, x: int, y: int, w: int, h: int, text: str) -> MarkupAnnotation:
        """Add rectangle markup around text"""
        padding = 10
        color = (0, 255, 0)  # Green
        thickness = 2
        
        cv2.rectangle(img, (x - padding, y - padding), 
                     (x + w + padding, y + h + padding), color, thickness)
        
        return MarkupAnnotation(
            markup_type='rectangle',
            subtype='standard',
            bbox=(x - padding, y - padding, w + 2 * padding, h + 2 * padding),
            text=text
        )
    
    def _add_variations(self, img: np.ndarray) -> np.ndarray:
        """Add realistic variations (noise, blur, brightness)"""
        # Random brightness
        if random.random() > 0.5:
            alpha = random.uniform(0.9, 1.1)
            beta = random.randint(-10, 10)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
        # Random slight blur
        if random.random() > 0.7:
            kernel_size = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        
        # Random noise
        if random.random() > 0.6:
            noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
            img = cv2.add(img, noise)
        
        return img
    
    def _save_manifest(self, samples: List[DocumentSample]):
        """Save a manifest file with all samples"""
        manifest = {
            'total_samples': len(samples),
            'languages': list(set(s.metadata['language'] for s in samples)),
            'samples': [
                {
                    'image_path': s.image_path,
                    'num_annotations': len(s.annotations),
                    'metadata': s.metadata
                }
                for s in samples
            ]
        }
        
        manifest_path = self.output_dir / 'manifest.json'
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        print(f"\nManifest saved to {manifest_path}")


if __name__ == "__main__":
    # Generate synthetic data
    generator = SyntheticDataGenerator(output_dir="data/synthetic_v2")
    
    print("Generating 1000 synthetic document samples...")
    samples = generator.generate_batch(
        count=1000,
        markup_types=['highlight', 'underline', 'strikethrough', 'circle', 'rectangle'],
        max_markups_per_image=3
    )
    
    print(f"\n✅ Generated {len(samples)} samples")
    print(f"Output directory: data/synthetic_v2/")
