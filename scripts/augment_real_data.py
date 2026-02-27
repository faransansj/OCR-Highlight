import os
import json
import cv2
import numpy as np
import sys
from tqdm import tqdm
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_generator.data_augmentation import DataAugmentation
from data_generation.yolo_converter import YOLODatasetConverter

def augment_real_data(num_aug=10):
    print(f"🚀 Augmenting Real Data (DocVQA) - target {num_aug}x per image")
    
    script_dir = Path(__file__).parent.parent
    input_dir = script_dir / "data" / "docvqa_with_markups"
    output_dir = script_dir / "data" / "docvqa_augmented"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Checking directory: {input_dir.absolute()}")
    
    augmentor = DataAugmentation(preserve_bbox=True)
    
    # Find all samples
    images = list(input_dir.glob("*.png"))
    print(f"Found {len(images)} base real images.")
    
    # Skip augmentation if already exists (to save time if re-running)
    existing_pngs = list(output_dir.glob("*.png"))
    if len(existing_pngs) >= 1000:
        print(f"Skipping augmentation, {len(existing_pngs)} images already exist.")
    else:
        aug_count = 0
        for img_path in tqdm(images, desc="Augmenting Real Data"):
            json_path = img_path.with_suffix(".json")
            if not json_path.exists(): continue
            
            # Load image and annotations
            image = cv2.imread(str(img_path))
            with open(json_path, 'r') as f:
                gt_data = json.load(f)
                
            # Extract bboxes and types for augmentation
            annots = gt_data.get('annotations', [])
            bboxes = [a['bbox'] for a in annots]
            types = [a['markup_type'] for a in annots]
            
            # Save original copy to augmented dir too
            cv2.imwrite(str(output_dir / img_path.name), image)
            with open(output_dir / json_path.name, 'w') as f:
                json.dump(gt_data, f, indent=2)
                
            # Generate augmentations
            for i in range(num_aug):
                try:
                    aug_img, aug_bboxes, aug_types = augmentor.augment_image(image, bboxes, types)
                except ValueError as e:
                    print(f"  Warning: Bbox error in {img_path.name}, skipping one variant: {e}")
                    continue
                
                aug_name = f"{img_path.stem}_aug{i}"
                aug_img_path = output_dir / f"{aug_name}.png"
                aug_json_path = output_dir / f"{aug_name}.json"
                
                cv2.imwrite(str(aug_img_path), aug_img)
                
                # Create new annotation JSON
                new_annots = []
                for bbox, mtype, orig_annot in zip(aug_bboxes, aug_types, annots):
                    new_annot = orig_annot.copy()
                    new_annot['bbox'] = list(bbox)
                    new_annots.append(new_annot)
                    
                aug_gt_data = gt_data.copy()
                aug_gt_data['annotations'] = new_annots
                aug_gt_data['output_image'] = str(aug_img_path)
                
                with open(aug_json_path, 'w') as f:
                    json.dump(aug_gt_data, f, indent=2)
                
                aug_count += 1
        print(f"\n✅ Created {aug_count} augmented samples in {output_dir}")

    # Convert to YOLO format
    # FIX: Use script_dir instead of base_dir
    yolo_output_dir = script_dir / "data" / "yolo_dataset_real_aug"
    converter = YOLODatasetConverter(output_dir=str(yolo_output_dir))
    
    print(f"Converting augmented dataset to YOLO format...")
    stats = converter.process_dataset(str(output_dir), val_split=0.15)
    
    print("\n✅ YOLO Dataset Ready (Augmented Real Data)")
    print(f"Train images: {stats['train']['images']}")
    print(f"Val images:   {stats['val']['images']}")

if __name__ == "__main__":
    augment_real_data(10)
