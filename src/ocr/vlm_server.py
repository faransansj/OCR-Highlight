import os
import io
import base64
import logging
from typing import List, Dict
import numpy as np
import cv2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VLMServer")

app = FastAPI(title="PaddleOCR-VL-1.5 Inference Server")

# Global engine placeholder
vlm_engine = None

class OCRRequest(BaseModel):
    image: str  # Base64 encoded image
    task: str = "spotting"
    lang: str = "en"

class OCRResponse(BaseModel):
    results: List[Dict]

@app.on_event("startup")
async def load_model():
    global vlm_engine
    try:
        from src.ocr.paddleocr_vl import PaddleOCRVL
        logger.info("Initializing PaddleOCR-VL-1.5 Engine on Server...")
        # Server typically has GPU access
        vlm_engine = PaddleOCRVL(use_gpu=True)
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize VLM Engine: {e}")

@app.post("/v1/ocr", response_model=OCRResponse)
async def perform_ocr(request: OCRRequest):
    if vlm_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # 1. Decode Base64 image
        img_data = base64.b64decode(request.image)
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        # 2. Run Inference
        results = vlm_engine.extract_text(image, lang=request.lang)
        
        return OCRResponse(results=results)

    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
