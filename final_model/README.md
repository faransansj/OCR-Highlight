# Markup Detection Model v1

## Model Info
- **Architecture**: YOLOv8n (nano)
- **Parameters**: 3,011,823
- **Input Size**: 640x640
- **Classes**: 5 (highlight, underline, strikethrough, circle, rectangle)

## Performance (Validation Set)
- **mAP50**: 96.10%
- **mAP50-95**: 75.44%
- **Precision**: 98.80%
- **Recall**: 63.31%

## Training Details
- **Dataset**: 9,090 train / 1,010 val images
- **Epochs**: 1 (early checkpoint)
- **Device**: CPU (AMD Ryzen 7 255)
- **Date**: 2026-02-07

## Files
- `markup_detector_v1.pt` - Best model weights
- `config.yaml` - Dataset configuration

## Usage
```python
from ultralytics import YOLO

model = YOLO('final_model/markup_detector_v1.pt')
results = model.predict('image.jpg')
```

## Notes
- High precision (98.8%) means very few false positives
- Lower recall (63.3%) means conservative detection
- Suitable for production use where accuracy > coverage
