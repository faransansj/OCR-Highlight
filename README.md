# 🖍️ OCR Highlight Extraction System

**Automatic markup detection and text extraction with high precision.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8n-orange.svg)](https://docs.ultralytics.com/)
[![mAP50](https://img.shields.io/badge/mAP50-96.1%25-success.svg)](docs/PERFORMANCE.md)

---

## 📖 Table of Contents
1.  [✨ Features](#-features)
2.  [🚀 Quick Start](#-quick-start)
3.  [🎯 YOLO Detection](#-yolo-detection)
4.  [📚 Documentation Hub](#-documentation-hub)
5.  [📂 Repository Map](#-repository-map)

---

## ✨ Features

-   ✅ **Deep Learning Detection**: YOLOv8 model for high-precision markup detection.
-   ✅ **Latest Model**: Trained on Apple Silicon M4 (`markup_detector_v2_m4.pt`).
-   ✅ **Class Consolidation**: Optimized for `highlight` and `symbol` detection.
-   ✅ **Multi-Language OCR**: Optimized for Korean-English mixed text.
-   ✅ **Universal Support**: Native acceleration for Apple Silicon (MPS), NVIDIA (CUDA), and AMD (ROCm).
-   ✅ **Speed**: Real-time inference (~40ms on CPU).

---

## 🚀 Quick Start

### Installation

```bash
# 1. Install uv (Fast package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Setup environment
uv venv && source .venv/bin/activate
uv pip install torch torchvision ultralytics opencv-python
```

### Inference (Run Prediction)

```bash
# Detect markup on an image
python3 predict.py path/to/image.jpg --conf 0.25
```

---

## 🎯 YOLO Detection

The new YOLO-based pipeline replaces the legacy HSV methods for better reliability.

### Platform-Specific Training
| Platform | Quick Command | Full Guide |
| :--- | :--- | :--- |
| **🍎 Mac (M1-M4)** | `uv run python train_yolo_mps.py` | [M4 Guide](docs/TRAINING.md) |
| **💻 NVIDIA GPU** | `uv run yolo train ... device=0` | [CUDA Guide](docs/TRAINING.md) |
| **🐧 AMD GPU** | `HSA_OVERRIDE_GFX_VERSION=11.0.2 uv run yolo ...` | [ROCm Guide](docs/TRAINING.md) |

---

## 📚 Documentation Hub

Explore detailed documentation for technical specifics:

-   [🎯 **Training Guide**](docs/TRAINING.md): Setup and training for all platforms.
-   [🏗️ **Architecture**](docs/ARCHITECTURE.md): System design and implementation details.
-   [📊 **Performance**](docs/PERFORMANCE.md): Detailed metrics and improvement history.
-   [📔 **Project Log**](docs/PROJECT_LOG.md): Current status and development timeline.

---

## 📂 Repository Map

```text
ocr-highlight-v2/
├── 📄 predict.py           # Main inference script
├── 📂 final_model/         # Best model weights (mAP 96%)
├── 📂 docs/                # Detailed documentation
├── 📂 src/                 # Core source code
├── 📂 data/                # Dataset folder
└── 📂 scripts/             # Utility scripts
```

---

## 🗺️ Roadmap (Future Phases)

### Phase 3: [차원 융합: 통합 파이프라인] (Current Focus)
**"SOTA OCR 엔진과 전용 마크업 탐지 모델의 결합"**
- **목표**: PaddleOCR-VL-1.5와 YOLOv8 마크업 모델의 통합 파이프라인 구축.
- **핵심 과제**:
    - [ ] **SOTA Engine Integration**: PaddleOCR-VL-1.5(텍스트/도표 추출) 연동.
    - [ ] **Markup-Text Fusion**: YOLOv8 탐지 좌표와 OCR 텍스트 좌표 매칭 알고리즘 구현.
    - [ ] **Metadata Enrichment**: 추출된 텍스트에 하이라이트, 밑줄 등 메타데이터 태깅 자동화.
    - [ ] **JSON/Markdown Export**: 구조화된 문서 데이터 출력 시스템 완성.

---

## 📄 License
MIT License. See [LICENSE](LICENSE) for details.

**Status**: ✅ Production-ready | **Version**: 2.0.0
