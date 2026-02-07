# 🎯 YOLO Training Guide

This guide provides step-by-step instructions for training the Markup Detection model on various platforms using `uv` (fast Python package manager) or standard environments.

---

## ⚡ Quick Setup with `uv` (Recommended)

`uv` is an extremely fast Python package installer and resolver.

```bash
# 1. Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Setup project environment
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
uv pip install torch torchvision torchaudio ultralytics
```

---

## 🚀 Running Training by Platform

### 🍎 Apple Silicon (M1, M2, M3, M4)
Uses Metal Performance Shaders (MPS) for acceleration.

```bash
# Option 1: Using the specialized script (recommended)
uv run python train_yolo_mps.py

# Option 2: Direct CLI
uv run yolo train data=data/yolo_dataset_preprocessed/dataset.yaml model=yolov8n.pt epochs=50 device=mps
```

### 💻 NVIDIA GPU (Windows/Linux)
Uses CUDA for acceleration.

```bash
uv run yolo train data=data/yolo_dataset_preprocessed/dataset.yaml model=yolov8n.pt epochs=50 device=0
```

### 🐧 AMD GPU (Linux/ROCm)
Uses ROCm. For certain Radeon 700M series (like 780M), spoofing might be required.

```bash
# For 780M (gfx1103) spoofing as gfx1102
HSA_OVERRIDE_GFX_VERSION=11.0.2 uv run yolo train data=data/yolo_dataset_preprocessed/dataset.yaml model=yolov8n.pt epochs=50 device=0
```

### 🖥️ CPU Only (Universal)
Slow but reliable for small tests.

```bash
uv run python train_yolo_cpu_final.py
```

---

## 📊 Expected Performance

| Device | Batch Size | Speed (it/s) | Est. Time (50 epochs) |
| :--- | :--- | :--- | :--- |
| **M4 (Apple Silicon)** | 16 | ~5-10 | **2-4 hours** |
| NVIDIA RTX 3060+ | 32 | ~20+ | < 1 hour |
| CPU (Modern i7/Ryzen 7) | 4 | ~1.6 | 10+ hours |

---

## 🔍 Verification

Check if your acceleration backend is active:

```bash
# Check MPS (Mac)
python3 -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

# Check CUDA (NVIDIA)
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 🛠️ Troubleshooting

- **Out of Memory (OOM):** Reduce `batch` size (e.g., `batch=8` or `batch=4`).
- **MPS Issues:** Ensure macOS 12.3+ and latest PyTorch.
- **Data Not Found:** Check `data/yolo_dataset_preprocessed/dataset.yaml` paths.

---

[⬅ Back to README](../README.md)
