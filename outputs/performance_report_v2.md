
# Performance Profiling Report
- **Hardware**: CPU Only (Local Simulation)
- **Image**: marked_docvqa_0019.png (1653x2339)

| Stage | Duration (s) | Notes |
| :--- | :--- | :--- |
| Initialization | 0.0160 | Loading ONNX model & engines |
| Detection | 0.2334 | YOLOv8 Nano ONNX @ 1280px |
| OCR (Crop) | 11.3021 | Sequential small region OCR |
| OCR (Fusion) | 21.6334 | Full-page OCR + Bbox matching |

## Analysis
- **Fusion mode** on CPU is significantly slower than **Crop mode** because full-page OCR is computationally expensive.
- **Projected Remote GPU (A6000)**: Fusion mode is expected to drop to **< 0.5s** total when using VLM API.
