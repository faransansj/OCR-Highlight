import cv2
import os
import json
from src.unified_pipeline_v2 import UnifiedPipelineV2

def save_failures():
    base_dir = "ocr-highlight-v2"
    with open(os.path.join(base_dir, "data/synthetic/base_annotations.json"), 'r') as f:
        data = json.load(f)
    
    pipeline = UnifiedPipelineV2(model_path=os.path.join(base_dir, "final_model/markup_detector_v1.pt"))
    
    os.makedirs("failure_crops", exist_ok=True)
    
    for i in range(2): # Just first 2 images
        img_path = os.path.join(base_dir, "data/synthetic", data[i]['image_name'])
        res = pipeline.process_image(img_path)
        img = cv2.imread(img_path)
        
        for j, r in enumerate(res['results']):
            x, y, w, h = r['bbox']
            crop = img[max(0,y):y+h, max(0,x):x+w]
            if crop.size > 0:
                cv2.imwrite(f"failure_crops/img{i}_crop{j}_{r['text']}.png", crop)

if __name__ == "__main__":
    save_failures()
