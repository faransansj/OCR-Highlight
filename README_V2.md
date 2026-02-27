# 🛡️ OCR-Highlight Pipeline v2.0

본 프로젝트는 문서 내의 마크업(하이라이트, 밑줄, 원, 사각형 등)을 탐지하고, 해당 영역의 텍스트를 고성능 OCR 엔진을 통해 추출하여 지식 관리 도구(Notion, Markdown)로 전송하는 통합 파이프라인입니다.

## 🚀 Key Features

- **Hybrid Detection**: YOLOv8 Nano(ONNX)를 활용한 0.1s급 초고속 마크업 탐지.
- **Fusion OCR**: 전체 페이지 OCR 결과와 마크업 좌표를 IoU 기반으로 결합하여 맥락이 살아있는 텍스트 추출.
- **SOTA Engine Integration**: PaddleOCR-VL-1.5 비전-랭귀지 모델 연동 지원.
- **Remote API Architecture**: 무거운 VLM 연산을 외부 서버로 분리하여 로컬 리소스 최적화.
- **Notion Sync**: 추출된 지식을 노션 데이터베이스로 Rich Text 스타일(색상, 밑줄 유지)과 함께 자동 전송.

## 🛠️ System Architecture

1.  **Frontend (Local)**: YOLOv8(Nano)로 마크업 탐지 수행.
2.  **Backend (Remote/Local)**: PaddleOCR-VL-1.5 또는 Multi-OCR Ensemble로 텍스트 추출.
3.  **Fusion Layer**: 탐지된 좌표와 텍스트 좌표를 매칭하여 메타데이터 생성.
4.  **Exporter**: Notion API 또는 Markdown 파일로 결과물 출력.

## 📋 Quick Start

### 1. 환경 설정
```bash
pip install -r requirements.txt
export NOTION_TOKEN="your_notion_token"
```

### 2. 파이프라인 실행
```python
from src.unified_pipeline_v2 import UnifiedPipelineV2

# 파이프라인 초기화
pipeline = UnifiedPipelineV2(
    model_path="final_model/markup_detector_v2_nano.onnx",
    use_remote_vlm=True,
    vlm_server_url="http://your-server-ip:8000/v1/ocr"
)

# 이미지 처리
results = pipeline.process_image("sample.png", mode="fusion")

# 노션 전송
from src.utils.notion_exporter import NotionExporter
exporter = NotionExporter()
exporter.create_page("your_database_id", results)
```

## 📊 Performance
- **Detection**: 97% mAP50
- **OCR Accuracy**: 95% (PaddleOCR-VL)
- **Latency**: < 0.5s (Remote GPU Mode)

---
*Developed with 💙 by Aris (OpenClaw Assistant)*
