# 🏗️ System Architecture

This document describes the technical implementation and design of the OCR Highlight Extraction System.

---

## 🛠️ Pipeline Overview

The system operates in two main modes:
1. **YOLO-based Markup Detection** (Deep Learning)
2. **HSV-based Color Segmentation** (Legacy / Classical CV)

---

## 🧠 Deep Learning Pipeline (YOLOv8)

Used for high-precision detection of 5 classes of markup.

### Detection Classes
0. `highlight`: Fluorescent markers
1. `underline`: Line under text
2. `strikethrough`: Line through text
3. `circle`: Circular annotation
4. `rectangle`: Box around text

### Data Preprocessing
Images are resized to **640px (max dimension)** to balance accuracy and memory consumption.

---

## 🏛️ Legacy Pipeline (HSV + Tesseract)

### 1. Highlight Detection
Utilizes HSL/HSV color space for robust segmentation under different lighting.

**HSV Ranges (`configs/optimized_hsv_ranges.json`):**
- **Yellow**: `[25, 60, 70]` to `[35, 255, 255]`
- **Green**: `[55, 60, 70]` to `[65, 255, 255]`
- **Pink**: `[169, 10, 70]` to `[180, 70, 255]`

### 2. Text Extraction (OCR)
- **Engine**: Tesseract LSTM
- **Mode**: Single line detection (PSM 7) with fallback logic.
- **Korean Optimization**: Recursive space removal and particle restoration.

---

## 📁 Repository Structure

```text
ocr-highlight-v2/
├── README.md               # 🏠 Main Entry & Quick Start
├── docs/                   # 📚 Detailed Documentation
│   ├── TRAINING.md         # 🎯 Platforms & Training Guide
│   ├── ARCHITECTURE.md     # 🏗️ Technical Implementation
│   ├── PERFORMANCE.md      # 📊 Metrics & Accuracy
│   └── PROJECT_LOG.md      # 📔 Status & Timeline
├── src/                    # 🧠 Core Modules
│   ├── data_generation/    # Synthetic generators
│   └── ocr/                # OCR engines
├── scripts/                # 🛠️ Utility & Debug Scripts
├── data/                   # 📂 Training & Preprocessed Data
├── final_model/            # 🏆 Released Model Weights
├── predict.py              # 🔍 Universal Inference (YOLO)
└── extract_highlights.py   # 🏛️ Legacy Extraction (HSV)
```

---

[⬅ Back to README](../README.md)
