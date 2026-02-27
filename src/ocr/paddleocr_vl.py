import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2

logger = logging.getLogger(__name__)

class PaddleOCRVL:
    """
    Wrapper for PaddleOCR-VL-1.5-0.9B Model.
    Designed for SOTA document parsing and text spotting.
    """
    def __init__(self, model_id: str = "PaddlePaddle/PaddleOCR-VL-1.5", use_gpu: bool = False):
        self.model_id = model_id
        self.use_gpu = use_gpu
        self.model = None
        self.processor = None
        logger.info(f"Initialized PaddleOCR-VL Wrapper for {model_id} (GPU: {use_gpu})")
        # Note: Actual model loading is deferred to allow fallback to other engines if transformers is missing.

    def load_model(self):
        """Lazy load the transformers model."""
        if self.model is not None:
            return

        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
            import torch

            device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
            logger.info(f"Loading PaddleOCR-VL-1.5 on {device}...")
            
            # This is a placeholder for the actual transformers loading logic for PaddleOCR-VL-1.5
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                trust_remote_code=True, 
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)
            
            logger.info("PaddleOCR-VL-1.5 loaded successfully.")
        except ImportError as e:
            logger.error(f"Failed to load transformers or torch: {e}")
            raise

    def extract_text(self, image: np.ndarray, lang: str = 'en') -> List[Dict]:
        """
        Extract text and bounding boxes using PaddleOCR-VL-1.5
        task "ocr" supports element-level recognition and text spotting.
        """
        self.load_model()
        
        from PIL import Image
        import torch
        
        # Convert BGR (OpenCV) to RGB (PIL)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        orig_w, orig_h = pil_image.size
        spotting_upscale_threshold = 1500
        task = "spotting" # We need spotting (text + bbox) for Fusion
        
        if orig_w < spotting_upscale_threshold and orig_h < spotting_upscale_threshold:
            process_w, process_h = orig_w * 2, orig_h * 2
            try:
                resample_filter = Image.Resampling.LANCZOS
            except AttributeError:
                resample_filter = Image.LANCZOS
            pil_image = pil_image.resize((process_w, process_h), resample_filter)

        max_pixels = 2048 * 28 * 28
        
        PROMPTS = {
            "ocr": "OCR:",
            "table": "Table Recognition:",
            "formula": "Formula Recognition:",
            "chart": "Chart Recognition:",
            "spotting": "Spotting:",
            "seal": "Seal Recognition:",
        }
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": PROMPTS[task]},
                ]
            }
        ]
        
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            images_kwargs={"size": {"shortest_edge": self.processor.image_processor.min_pixels, "longest_edge": max_pixels}},
        ).to(self.model.device)
        
        outputs = self.model.generate(**inputs, max_new_tokens=2048)
        result_text = self.processor.decode(outputs[0][inputs["input_ids"].shape[-1]:-1])
        
        logger.info(f"PaddleOCR-VL-1.5 Raw Output (length: {len(result_text)})")
        
        # Parse result_text
        # Typical spotting format: "text_content<bbox>[[x1, y1, x2, y2]]</bbox>" or similar JSON
        import re
        results = []
        
        # Heuristic parsing for common VLM spotting formats
        # Pattern 1: Look for JSON-like lists [[x1, y1, x2, y2]]
        # We try to extract text and the bounding box following it.
        pattern = re.compile(r'(.*?)(?:<bbox>|<box>)?\s*\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\](?:</bbox>|</box>)?')
        matches = pattern.findall(result_text)
        
        if matches:
            for match in matches:
                text_content = match[0].strip()
                if not text_content: continue
                
                # Coordinates are often normalized (0-1000) or absolute. We assume absolute or scaled to max_pixels.
                # Assuming absolute coordinates in original image scale for this wrapper.
                x1, y1, x2, y2 = map(int, match[1:5])
                
                # If they are normalized (e.g. 0-1000):
                if x2 <= 1000 and y2 <= 1000 and orig_w > 1000:
                    x1 = int(x1 / 1000.0 * orig_w)
                    y1 = int(y1 / 1000.0 * orig_h)
                    x2 = int(x2 / 1000.0 * orig_w)
                    y2 = int(y2 / 1000.0 * orig_h)
                
                # To [x, y, w, h]
                w = x2 - x1
                h = y2 - y1
                
                results.append({
                    'text': text_content,
                    'bbox': [x1, y1, w, h],
                    'conf': 0.95 # VLMs typically don't output per-word confidence, so we assign a high default
                })
        else:
            # Fallback if the format is purely text without boxes (due to error or different prompt)
            # We just return the whole text as one block
            if result_text.strip():
                results.append({
                    'text': result_text.strip(),
                    'bbox': [0, 0, orig_w, orig_h],
                    'conf': 0.8
                })
                
        return results

if __name__ == "__main__":
    # Test initialization
    vl_engine = PaddleOCRVL()
    print("PaddleOCR-VL-1.5 wrapper initialized.")
