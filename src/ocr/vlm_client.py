import os
import io
import json
import base64
import logging
import asyncio
import aiohttp
from typing import List, Dict, Optional
import numpy as np
import cv2

logger = logging.getLogger(__name__)

class PaddleOCRVLClient:
    """
    Asynchronous Client for communicating with the external PaddleOCR-VL-1.5 Server.
    """
    def __init__(self, api_url: str = "http://localhost:8000/v1/ocr", timeout: int = 30):
        self.api_url = api_url
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        logger.info(f"Initialized PaddleOCR-VL Client pointing to {self.api_url}")

    def _encode_image(self, image: np.ndarray) -> str:
        """Convert OpenCV image to base64 string"""
        _, buffer = cv2.imencode('.png', image)
        return base64.b64encode(buffer).decode('utf-8')

    async def fetch_ocr_results_async(self, image: np.ndarray, lang: str = 'en') -> List[Dict]:
        """
        Send image to the server and receive text spotting results.
        Expected server response: {'results': [{'text': '...', 'bbox': [x, y, w, h], 'conf': 0.9}, ...]}
        """
        base64_image = self._encode_image(image)
        
        payload = {
            "image": base64_image,
            "task": "spotting",
            "lang": lang
        }

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(self.api_url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('results', [])
                    else:
                        error_text = await response.text()
                        logger.error(f"API Error {response.status}: {error_text}")
                        return []
        except Exception as e:
            logger.error(f"Failed to connect to VLM server: {e}")
            return []

    def extract_text(self, image: np.ndarray, lang: str = 'en') -> List[Dict]:
        """
        Synchronous wrapper for the async API call.
        To be used as a drop-in replacement in the unified pipeline.
        """
        return asyncio.run(self.fetch_ocr_results_async(image, lang))

if __name__ == "__main__":
    # Simple test
    client = PaddleOCRVLClient()
    print("PaddleOCR-VL-1.5 Client initialized successfully.")
