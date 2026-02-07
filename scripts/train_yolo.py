"""
YOLOv8 Training Script for Document Markup Detection
"""

from ultralytics import YOLO
import torch
from pathlib import Path
import json
from datetime import datetime


def train_markup_detector(
    data_yaml: str = 'data/yolo_dataset/dataset.yaml',
    model_size: str = 'n',  # n, s, m, l, x
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    patience: int = 15,
    device: str = 'cpu',
    project: str = 'runs/detect',
    name: str = 'markup_detector'
):
    """
    Train YOLOv8 model for markup detection
    
    Args:
        data_yaml: Path to dataset YAML config
        model_size: Model size (n=nano, s=small, m=medium, l=large, x=xlarge)
        epochs: Number of training epochs
        imgsz: Input image size
        batch: Batch size
        patience: Early stopping patience
        device: Device to use ('cpu', 'cuda', or device number)
        project: Project directory
        name: Experiment name
    """
    
    print("="*60)
    print("YOLOv8 Training for Document Markup Detection")
    print("="*60)
    print()
    
    # Check device
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA not available, falling back to CPU")
        device = 'cpu'
    
    print(f"Device: {device}")
    print(f"Model: YOLOv8{model_size}")
    print(f"Data: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    print()
    
    # Load pretrained model
    model_name = f'yolov8{model_size}.pt'
    print(f"Loading pretrained model: {model_name}...")
    model = YOLO(model_name)
    
    # Start training
    print()
    print("🚀 Starting training...")
    print()
    
    start_time = datetime.now()
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        save=True,
        device=device,
        project=project,
        name=name,
        verbose=True,
        plots=True,
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        mosaic=1.0,
        # Optimization
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
    )
    
    end_time = datetime.now()
    training_time = end_time - start_time
    
    print()
    print("="*60)
    print(f"✅ Training completed in {training_time}")
    print("="*60)
    print()
    
    # Save training summary
    summary = {
        'model': model_name,
        'data': data_yaml,
        'epochs': epochs,
        'training_time': str(training_time),
        'device': device,
        'final_metrics': {
            'mAP50': float(results.results_dict.get('metrics/mAP50(B)', 0)),
            'mAP50-95': float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
            'precision': float(results.results_dict.get('metrics/precision(B)', 0)),
            'recall': float(results.results_dict.get('metrics/recall(B)', 0)),
        },
        'best_model': str(Path(project) / name / 'weights' / 'best.pt')
    }
    
    summary_path = Path(project) / name / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Training summary saved to: {summary_path}")
    print()
    print("Final Metrics:")
    for metric, value in summary['final_metrics'].items():
        print(f"  {metric}: {value:.4f}")
    print()
    
    return model, results, summary


def validate_model(model_path: str, data_yaml: str):
    """
    Validate trained model on validation set
    """
    print("="*60)
    print("Model Validation")
    print("="*60)
    print()
    
    model = YOLO(model_path)
    
    results = model.val(data=data_yaml, verbose=True)
    
    print()
    print("Validation Results:")
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")
    print(f"  Precision: {results.box.p:.4f}")
    print(f"  Recall: {results.box.r:.4f}")
    print()
    
    return results


def test_inference(model_path: str, test_image: str):
    """
    Test inference on a single image
    """
    print(f"Testing inference on: {test_image}")
    
    model = YOLO(model_path)
    
    results = model(test_image, conf=0.25, iou=0.45)
    
    # Print detections
    for result in results:
        boxes = result.boxes
        print(f"  Detected {len(boxes)} markups:")
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"    - Class {cls} (conf: {conf:.3f})")
    
    # Save annotated image
    output_path = Path(test_image).parent / f"annotated_{Path(test_image).name}"
    results[0].save(str(output_path))
    print(f"  Saved annotated image: {output_path}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLOv8 for markup detection')
    parser.add_argument('--data', default='data/yolo_dataset/dataset.yaml', help='Dataset YAML')
    parser.add_argument('--model', default='n', choices=['n', 's', 'm', 'l', 'x'], help='Model size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--device', default='cpu', help='Device (cpu or cuda)')
    parser.add_argument('--validate', action='store_true', help='Run validation after training')
    parser.add_argument('--test-image', help='Test image path for inference')
    
    args = parser.parse_args()
    
    # Train
    model, results, summary = train_markup_detector(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device
    )
    
    # Validate
    if args.validate:
        best_model = summary['best_model']
        validate_model(best_model, args.data)
    
    # Test inference
    if args.test_image:
        best_model = summary['best_model']
        test_inference(best_model, args.test_image)
