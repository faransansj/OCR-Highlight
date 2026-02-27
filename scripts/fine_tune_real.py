import os
import sys
import json
from pathlib import Path
from ultralytics import YOLO
import torch
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

def fine_tune_on_real_data():
    print("\n" + "="*70)
    print("🚀 MILESTONE 2: REAL DATA FINE-TUNING")
    print("="*70 + "\n")

    # Paths (relative to script location)
    script_dir = Path(__file__).parent.parent
    base_model_path = script_dir / "final_model" / "markup_detector_v2_m4.pt"
    # Using augmented real data
    data_yaml = script_dir / "data" / "yolo_dataset_real_aug" / "dataset.yaml"
    
    if not base_model_path.exists():
        print(f"Error: Base model not found at {base_model_path.absolute()}")
        return
    
    if not data_yaml.exists():
        print(f"Error: Dataset YAML not found at {data_yaml.absolute()}")
        return

    # Device configuration
    # Forcing CPU due to known ROCm incompatibility with gfx1103 for complex CNN ops
    device = 'cpu'
    print(f"Using device: {device}")

    # Load model
    print(f"Loading base model: {base_model_path}")
    model = YOLO(base_model_path)

    # Fine-tuning Parameters
    # Optimized for CPU stability in sandbox environment
    epochs = 15
    batch_size = 4
    imgsz = 640 

    print(f"Starting optimized fine-tuning for {epochs} epochs...")
    
    start_time = datetime.now()
    
    results = model.train(
        data=str(data_yaml.absolute()),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project=str((script_dir / "runs/fine_tune").absolute()),
        name="real_world_optimized_v2",
        lr0=0.0001,
        mosaic=0.3,
        mixup=0.0,
        save=True,
        workers=2 # Limited workers to prevent resource exhaustion
    )
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n✅ Fine-tuning completed in {duration}")
    
    # Save best weights
    best_weights = Path(results.save_dir) / 'weights' / 'best.pt'
    final_model_dir = script_dir / "final_model"
    final_model_dir.mkdir(exist_ok=True)
    
    target_path = final_model_dir / "markup_detector_v2_real_aug.pt"
    import shutil
    shutil.copy(best_weights, target_path)
    
    print(f"✅ Final model saved to: {target_path}")

if __name__ == "__main__":
    fine_tune_on_real_data()
