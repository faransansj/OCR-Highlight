import os
import sys
import logging
import argparse
import time
from typing import Dict

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.unified_pipeline_v2 import UnifiedPipelineV2
from src.utils.notion_exporter import NotionExporter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("E2E-Test")

def run_e2e_test(image_path: str, notion_db_id: str = None, use_remote: bool = False):
    logger.info(f"Starting E2E Integration Test for image: {image_path}")
    
    # 1. Initialize Pipeline
    # Using local ONNX model for speed
    pipeline = UnifiedPipelineV2(
        model_path="final_model/markup_detector_v2_nano.onnx",
        use_remote_vlm=use_remote
    )
    
    # 2. Process Image
    start_time = time.time()
    logger.info("Step 1: Running detection and OCR fusion...")
    mode = "fusion" if use_remote else "crop"
    results = pipeline.process_image(image_path, mode=mode)
    process_time = time.time() - start_time
    
    logger.info(f"Step 1 Complete. Time: {process_time:.2f}s")
    logger.info(f"Detected {len(results['results'])} markup regions.")
    
    # 3. Metadata Enrichment Check
    logger.info("Step 2: Checking metadata enrichment...")
    if results.get('markdown'):
        logger.info("Markdown with enriched metadata generated successfully.")
    else:
        logger.warning("Markdown generation empty.")

    # 4. Notion Export (Dry Run if no DB ID)
    if notion_db_id:
        logger.info(f"Step 3: Exporting to Notion Database {notion_db_id}...")
        exporter = NotionExporter()
        export_res = exporter.create_page(notion_db_id, results)
        if export_res:
            logger.info("Step 3 Complete. Page created successfully.")
        else:
            logger.error("Step 3 Failed. Notion export failed.")
    else:
        logger.info("Step 3: Skipping Notion export (No Database ID provided).")

    logger.info("--- E2E Integration Test Finished ---")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="data/yolo_dataset_real/val/images/marked_docvqa_0019.png")
    parser.add_argument("--db_id", type=str, help="Notion Database ID for testing")
    parser.add_argument("--remote", action="store_true", help="Use remote VLM server")
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        logger.error(f"Image not found: {args.image}")
        sys.exit(1)
        
    run_e2e_test(args.image, args.db_id, args.remote)
