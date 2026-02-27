import sys
import os
import json
import time

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.unified_pipeline_v2 import UnifiedPipelineV2

def run_test():
    test_image = "test_results/generator_test/sample_000000_kor.png"
    if not os.path.exists(test_image):
        print(f"Test image not found: {test_image}")
        # Try a real image
        test_image = "data/yolo_dataset_real/val/images/marked_docvqa_0019.png"
        if not os.path.exists(test_image):
            print("No test images available.")
            return

    print(f"Testing Fusion Mode on: {test_image}")
    
    # Initialize pipeline
    # Use CPU for testing if GPU is unavailable/problematic
    pipeline = UnifiedPipelineV2(
        model_path="final_model/markup_detector_v2_nano.pt",
        ocr_engines=['easyocr'], # Use easyocr for faster test
        use_gpu=False
    )
    
    print("\n--- Running in Legacy CROP mode ---")
    start_time = time.time()
    res_crop = pipeline.process_image(test_image, mode="crop")
    crop_time = time.time() - start_time
    print(f"Time taken: {crop_time:.2f}s")
    print(f"Detected {len(res_crop['results'])} items.")
    for r in res_crop['results']:
        print(f"  [{r['markup_type']}] {r.get('text', '')[:30]}... (conf: {r.get('confidence_ocr', 0.0):.2f})")
        
    print("\n--- Running in New FUSION mode ---")
    start_time = time.time()
    res_fusion = pipeline.process_image(test_image, mode="fusion")
    fusion_time = time.time() - start_time
    print(f"Time taken: {fusion_time:.2f}s")
    print(f"Detected {len(res_fusion['results'])} items.")
    for r in res_fusion['results']:
        print(f"  [{r['markup_type']}] {r.get('text', '')[:30]}... (conf: {r.get('confidence_ocr', 0.0):.2f})")

if __name__ == "__main__":
    run_test()
