# Training Guide for Apple Silicon (M4)

## Prerequisites

### 1. Install `uv` and Setup Environment (Recommended)
`uv` is an extremely fast Python package installer and resolver.

```bash
# 1. Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create virtual environment
uv venv

# 3. Activate environment
source .venv/bin/activate

# 4. Install dependencies
uv pip install torch torchvision torchaudio ultralytics
```

### 2. Traditional Setup (Alternative)
```bash
python3 -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}'); print(f'MPS built: {torch.backends.mps.is_built()}')"
```

Expected output:
```
MPS available: True
MPS built: True
```

## Training Script for Apple Silicon

Create `train_yolo_mps.py`:

```python
#!/usr/bin/env python3
"""
YOLOv8 Training on Apple Silicon (M4) with MPS
"""
import torch
from ultralytics import YOLO

print("=" * 60)
print("🍎 Apple Silicon (M4) YOLO Training")
print("=" * 60)
print(f"PyTorch: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
print("=" * 60)

# Load pretrained model
model = YOLO('yolov8n.pt')

# Train with MPS (Apple Silicon GPU)
results = model.train(
    data='data/yolo_dataset_preprocessed/dataset.yaml',
    epochs=50,
    imgsz=640,
    batch=16,          # Adjust based on available memory
    device='mps',      # Use Apple Metal Performance Shaders
    project='markup_detector_mps',
    name='m4_training',
    patience=10,
    save=True,
    save_period=5,     # Save checkpoint every 5 epochs
    cache=False,
    workers=4,
    amp=False,         # AMP not needed on MPS
    verbose=True
)

print("\n🎉 Training completed!")
print(f"Best model: markup_detector_mps/m4_training/weights/best.pt")
```

## Running Training

### Option 1: Using `uv run` (Fastest)
```bash
uv run python train_yolo_mps.py
```

### Option 2: Direct Run (Activated Env)
```bash
python3 train_yolo_mps.py 2>&1 | tee training_log.txt
```

## Expected Performance

**On M4 (Estimated):**
- Speed: ~5-10x faster than CPU
- Batch size 16: ~3-5 seconds/epoch (depends on dataset size)
- Total time (50 epochs): ~2-4 hours
- Memory usage: ~8-12GB unified memory

## Monitoring

### Check GPU Usage
```bash
# Monitor GPU activity
sudo powermetrics --samplers gpu_power -i 1000
```

### TensorBoard (Optional)
```bash
tensorboard --logdir runs/detect
```

## Troubleshooting

### Issue: MPS not available
**Solution:**
- Update to latest PyTorch: `pip install --upgrade torch torchvision`
- Ensure macOS 12.3+ (Monterey or later)

### Issue: Out of memory
**Solution:**
- Reduce batch size: `batch=8` or `batch=4`
- Reduce image size: `imgsz=512`

### Issue: Training too slow
**Solution:**
- Verify MPS is being used: Check logs for `device=mps`
- Increase batch size if memory allows
- Enable `cache=True` (if enough RAM)

## Model Export for Deployment

After training, export to different formats:

```python
from ultralytics import YOLO

# Load best model
model = YOLO('markup_detector_mps/m4_training/weights/best.pt')

# Export to CoreML (optimized for Apple devices)
model.export(format='coreml')

# Export to ONNX (cross-platform)
model.export(format='onnx')
```

## Performance Comparison

| Device | Batch Size | Speed (it/s) | Total Time (50 epochs) |
|--------|------------|--------------|------------------------|
| CPU (AMD Ryzen 7) | 4 | 1.6 | ~10+ hours |
| AMD Radeon 780M | - | Failed | ISA incompatibility |
| **M4 (MPS)** | 16 | ~5-10 | **2-4 hours** |

## Notes

- **MPS vs CUDA**: MPS is Apple's GPU framework, similar to CUDA for NVIDIA
- **Unified Memory**: M4 uses shared memory between CPU/GPU (no separate VRAM)
- **Batch Size**: Start with 16, increase if memory allows (M4 has plenty)
- **amp=False**: Mixed precision training not needed on Apple Silicon

## Next Steps

1. Run training on M4
2. Monitor performance and adjust batch size
3. Export best model for deployment
4. Test on validation set

Good luck! 🚀
