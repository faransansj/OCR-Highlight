"""
Convert our annotation format to YOLO format for training
"""

import json
from pathlib import Path
import shutil
from typing import List, Dict
import yaml


class YOLODatasetConverter:
    """
    Convert OCR-Highlight annotations to YOLO format
    """
    
    # Class mapping
    CLASS_MAP = {
        'highlight': 0,
        'underline': 1,
        'double_underline': 1,  # Treat as underline
        'strikethrough': 2,
        'circle': 3,
        'rectangle': 4
    }
    
    CLASS_NAMES = ['highlight', 'underline', 'strikethrough', 'circle', 'rectangle']
    
    def __init__(self, output_dir: str = 'data/yolo_dataset'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create train/val splits
        self.train_dir = self.output_dir / 'train'
        self.val_dir = self.output_dir / 'val'
        
        for split in [self.train_dir, self.val_dir]:
            (split / 'images').mkdir(parents=True, exist_ok=True)
            (split / 'labels').mkdir(parents=True, exist_ok=True)
    
    def bbox_to_yolo(self, bbox: List[int], img_width: int, img_height: int) -> List[float]:
        """
        Convert [x, y, w, h] to YOLO format [x_center, y_center, width, height] (normalized)
        """
        x, y, w, h = bbox
        
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        width = w / img_width
        height = h / img_height
        
        return [x_center, y_center, width, height]
    
    def convert_annotation(self, ann_data: Dict, img_width: int, img_height: int) -> List[str]:
        """
        Convert single annotation to YOLO format lines
        """
        lines = []
        
        for ann in ann_data.get('annotations', []):
            markup_type = ann['markup_type']
            
            if markup_type not in self.CLASS_MAP:
                continue
            
            class_id = self.CLASS_MAP[markup_type]
            bbox = ann['bbox']
            
            # Convert to YOLO format
            yolo_bbox = self.bbox_to_yolo(bbox, img_width, img_height)
            
            # YOLO format: class_id x_center y_center width height
            line = f"{class_id} {' '.join([f'{v:.6f}' for v in yolo_bbox])}"
            lines.append(line)
        
        return lines
    
    def process_dataset(self, image_dir: str, val_split: float = 0.1) -> Dict:
        """
        Process entire dataset and split into train/val
        
        Args:
            image_dir: Directory with images and JSON annotations
            val_split: Fraction for validation set
        
        Returns:
            Statistics dictionary
        """
        image_dir = Path(image_dir)
        
        # Find all images with annotations
        image_files = []
        for ext in ['*.png', '*.jpg']:
            for img_path in image_dir.glob(ext):
                json_path = img_path.with_suffix('.json')
                if json_path.exists():
                    image_files.append(img_path)
        
        print(f"Found {len(image_files)} images with annotations")
        
        # Shuffle and split
        import random
        random.seed(42)
        random.shuffle(image_files)
        
        val_count = int(len(image_files) * val_split)
        val_files = image_files[:val_count]
        train_files = image_files[val_count:]
        
        print(f"Train: {len(train_files)}, Val: {len(val_files)}")
        
        # Process train and val
        stats = {
            'train': self._process_split(train_files, self.train_dir),
            'val': self._process_split(val_files, self.val_dir)
        }
        
        # Create dataset YAML
        self._create_dataset_yaml()
        
        return stats
    
    def _process_split(self, image_files: List[Path], output_dir: Path) -> Dict:
        """Process a split (train or val)"""
        import cv2
        
        stats = {'images': 0, 'annotations': 0, 'classes': {name: 0 for name in self.CLASS_NAMES}}
        
        for img_path in image_files:
            json_path = img_path.with_suffix('.json')
            
            # Load annotation
            with open(json_path) as f:
                ann_data = json.load(f)
            
            # Get image size
            if 'image_size' in ann_data:
                # Real data format (DocVQA)
                img_width = ann_data['image_size']['width']
                img_height = ann_data['image_size']['height']
            else:
                # Synthetic data or fallback: read image
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"  Warning: Failed to load {img_path}")
                    continue
                img_height, img_width = img.shape[:2]
            
            # Convert to YOLO format
            yolo_lines = self.convert_annotation(ann_data, img_width, img_height)
            
            if not yolo_lines:
                continue
            
            # Copy image
            dest_img = output_dir / 'images' / img_path.name
            shutil.copy(img_path, dest_img)
            
            # Write label file
            label_file = output_dir / 'labels' / f"{img_path.stem}.txt"
            with open(label_file, 'w') as f:
                f.write('\n'.join(yolo_lines))
            
            # Update stats
            stats['images'] += 1
            stats['annotations'] += len(yolo_lines)
            
            for line in yolo_lines:
                class_id = int(line.split()[0])
                class_name = self.CLASS_NAMES[class_id]
                stats['classes'][class_name] += 1
        
        return stats
    
    def _create_dataset_yaml(self):
        """Create YOLO dataset configuration file"""
        config = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'nc': len(self.CLASS_NAMES),
            'names': self.CLASS_NAMES
        }
        
        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✅ Dataset YAML created: {yaml_path}")
    
    def merge_datasets(self, *dataset_dirs: str) -> Dict:
        """
        Merge multiple datasets (synthetic + real)
        
        Args:
            *dataset_dirs: Paths to dataset directories
        
        Returns:
            Combined statistics
        """
        all_images = []
        
        for dataset_dir in dataset_dirs:
            dataset_dir = Path(dataset_dir)
            for ext in ['*.png', '*.jpg']:
                for img_path in dataset_dir.glob(ext):
                    json_path = img_path.with_suffix('.json')
                    if json_path.exists():
                        all_images.append(img_path)
        
        print(f"Merging {len(all_images)} images from {len(dataset_dirs)} datasets")
        
        # Process merged dataset
        import random
        random.seed(42)
        random.shuffle(all_images)
        
        val_count = int(len(all_images) * 0.1)
        val_files = all_images[:val_count]
        train_files = all_images[val_count:]
        
        stats = {
            'train': self._process_split(train_files, self.train_dir),
            'val': self._process_split(val_files, self.val_dir)
        }
        
        self._create_dataset_yaml()
        
        return stats


if __name__ == '__main__':
    converter = YOLODatasetConverter()
    
    # Merge synthetic + real datasets
    stats = converter.merge_datasets(
        'data/synthetic_v2_large',  # 10K synthetic
        'data/docvqa_with_markups'  # 100 real
    )
    
    print("\n✅ YOLO Dataset Conversion Complete!")
    print(f"\nTrain Statistics:")
    print(f"  Images: {stats['train']['images']}")
    print(f"  Annotations: {stats['train']['annotations']}")
    print(f"  Classes:")
    for class_name, count in stats['train']['classes'].items():
        print(f"    - {class_name}: {count}")
    
    print(f"\nVal Statistics:")
    print(f"  Images: {stats['val']['images']}")
    print(f"  Annotations: {stats['val']['annotations']}")
    print(f"  Classes:")
    for class_name, count in stats['val']['classes'].items():
        print(f"    - {class_name}: {count}")
