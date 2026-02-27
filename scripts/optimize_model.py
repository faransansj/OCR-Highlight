import os
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Optimizer")

def export_model():
    model_path = "final_model/markup_detector_v2_nano.pt"
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return

    logger.info(f"Loading model for optimization: {model_path}")
    model = YOLO(model_path)

    # Export to ONNX format with FP16 or half precision if supported
    # This reduces VRAM usage and can speed up CPU inference via OpenVINO or ONNXRuntime
    try:
        logger.info("Exporting to ONNX format...")
        success_path = model.export(format="onnx", imgsz=1280, simplify=True)
        logger.info(f"Optimization complete! Exported to: {success_path}")
        
        # Also try OpenVINO for Intel/General CPU optimization if available
        # success_path_ov = model.export(format="openvino", imgsz=1280)
        # logger.info(f"OpenVINO export complete: {success_path_ov}")
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")

if __name__ == "__main__":
    export_model()
