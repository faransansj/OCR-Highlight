"""
Image Preprocessing for YOLOv8 Training
Resize and optimize images to reduce memory usage
"""

from PIL import Image
from pathlib import Path
from tqdm import tqdm
import shutil

def preprocess_images(
    source_dir: str,
    target_dir: str,
    max_size: int = 640,
    quality: int = 95
):
    """
    Preprocess images: resize and optimize
    
    Args:
        source_dir: Source directory with images
        target_dir: Target directory for processed images
        max_size: Maximum dimension (width or height)
        quality: JPEG quality (if converting)
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    image_files = list(source_path.glob("*.png")) + list(source_path.glob("*.jpg"))
    
    print(f"Processing {len(image_files)} images...")
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    print(f"Max size: {max_size}")
    print()
    
    total_original_size = 0
    total_new_size = 0
    
    for img_path in tqdm(image_files, desc="Preprocessing"):
        # Open image
        img = Image.open(img_path)
        original_size = img_path.stat().st_size
        total_original_size += original_size
        
        # Calculate new dimensions
        width, height = img.size
        if width > max_size or height > max_size:
            # Resize maintaining aspect ratio
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Save optimized image
        output_path = target_path / img_path.name
        img.save(output_path, "PNG", optimize=True)
        
        new_size = output_path.stat().st_size
        total_new_size += new_size
    
    # Calculate savings
    reduction_pct = (1 - total_new_size / total_original_size) * 100
    
    print()
    print("=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)
    print(f"Total original size: {total_original_size / (1024**2):.2f} MB")
    print(f"Total new size: {total_new_size / (1024**2):.2f} MB")
    print(f"Reduction: {reduction_pct:.1f}%")
    print()


def preprocess_labels(source_dir: str, target_dir: str):
    """
    Copy label files to target directory
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    label_files = list(source_path.glob("*.txt"))
    
    print(f"Copying {len(label_files)} label files...")
    
    for label_path in tqdm(label_files, desc="Copying labels"):
        shutil.copy2(label_path, target_path / label_path.name)
    
    print("Labels copied!")
    print()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess images for YOLOv8')
    parser.add_argument('--max-size', type=int, default=640, help='Max image dimension')
    parser.add_argument('--quality', type=int, default=95, help='Image quality')
    
    args = parser.parse_args()
    
    # Preprocess train images
    print("=" * 60)
    print("TRAIN SET")
    print("=" * 60)
    preprocess_images(
        'data/yolo_dataset/train/images',
        'data/yolo_dataset_preprocessed/train/images',
        max_size=args.max_size,
        quality=args.quality
    )
    
    preprocess_labels(
        'data/yolo_dataset/train/labels',
        'data/yolo_dataset_preprocessed/train/labels'
    )
    
    # Preprocess val images
    print("=" * 60)
    print("VALIDATION SET")
    print("=" * 60)
    preprocess_images(
        'data/yolo_dataset/val/images',
        'data/yolo_dataset_preprocessed/val/images',
        max_size=args.max_size,
        quality=args.quality
    )
    
    preprocess_labels(
        'data/yolo_dataset/val/labels',
        'data/yolo_dataset_preprocessed/val/labels'
    )
    
    # Create dataset.yaml with relative paths
    yaml_content = f"""names:
- highlight
- underline
- strikethrough
- circle
- rectangle
nc: 5
path: ./data/yolo_dataset_preprocessed
train: train/images
val: val/images
"""
    
    yaml_path = Path('data/yolo_dataset_preprocessed/dataset.yaml')
    yaml_path.write_text(yaml_content)
    
    print("=" * 60)
    print("✅ Preprocessing complete!")
    print(f"Dataset config: {yaml_path}")
    print("=" * 60)
