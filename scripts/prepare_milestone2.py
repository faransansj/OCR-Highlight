import os
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_generation.yolo_converter import YOLODatasetConverter

def prepare_real_data():
    print("🚀 Entering Milestone 2: [현실 세계로] - Phase 1: Real Data Preparation")
    
    # Define paths
    real_data_dir = 'data/docvqa_with_markups'
    yolo_output_dir = 'data/yolo_dataset_real'
    
    if not os.path.exists(real_data_dir):
        print(f"Error: {real_data_dir} not found.")
        return

    # Initialize Converter
    converter = YOLODatasetConverter(output_dir=yolo_output_dir)
    
    # Process dataset (Val split 20% for 100 samples)
    print(f"Converting {real_data_dir} to YOLO format...")
    stats = converter.process_dataset(real_data_dir, val_split=0.2)
    
    print("\n✅ Real Data Conversion Complete!")
    print(f"Train: {stats['train']['images']} images")
    print(f"Val:   {stats['val']['images']} images")
    
    print("\nClass distribution:")
    for cls, count in stats['train']['classes'].items():
        print(f"  {cls}: {count}")

    # Generate dataset.yaml for fine-tuning
    # (Converter already does this, but ensure path is correct for training)
    
if __name__ == "__main__":
    prepare_real_data()
