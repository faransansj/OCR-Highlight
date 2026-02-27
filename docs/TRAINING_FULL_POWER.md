# 🚀 Full Power Training Guide (GPU/Colab)

이 가이드는 구글 코랩(Colab)이나 랩실의 NVIDIA GPU 환경에서 아리스의 성능을 한계까지 끌어올리기 위한 **'초월 학습'** 매뉴얼입니다.

---

## 🏗️ 1. 환경 구축 (Environment Setup)

### Option A: Google Colab (추천)
코랩은 NVIDIA T4, V100, A100 등 강력한 GPU를 무료 또는 저렴하게 대여해줍니다.

1.  **런타임 유형 변경**: `런타임` > `런타임 유형 변경` > `T4 GPU` (또는 그 이상) 선택.
2.  **필수 라이브러리 설치**:
    ```python
    !pip install ultralytics
    ```
3.  **데이터셋 업로드**: 구글 드라이브를 마운트하여 데이터를 빠르게 로드합니다.
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

### Option B: 랩실 NVIDIA GPU (Local/Server)
1.  **CUDA 확인**: `nvidia-smi` 커맨드로 GPU 상태와 CUDA 버전을 확인합니다.
2.  **환경 생성 (uv 권장)**:
    ```bash
    uv venv
    source .venv/bin/activate
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # CUDA 버전에 맞춰 설치
    uv pip install ultralytics
    ```

---

## ⚡ 2. 초월 학습 스크립트 (`train_full_power.py`)

기존보다 더 높은 해상도와 더 큰 모델을 사용하여 '풀 파워'로 학습합니다.

```python
from ultralytics import YOLO

# 1. 모델 체급 상향 (Nano -> Small or Medium)
# 'yolov8s.pt' (Small)는 Nano보다 똑똑하지만 여전히 빠릅니다.
model = YOLO('yolov8s.pt') 

# 2. 초월 학습 시작
results = model.train(
    data='data/dataset.yaml',
    epochs=100,         # 100회 반복 학습
    imgsz=1024,         # 해상도를 1024로 높여 작은 기호 정밀도 향상
    batch=-1,           # GPU 메모리에 맞춰 자동 조절 (Auto-batch)
    device=0,           # NVIDIA GPU 0번 사용
    project='markup_detector_full_power',
    name='v2_small_model',
    save=True,
    amp=True,           # 가속 학습(Mixed Precision) 활성화
    mosaic=1.0,         # 데이터 증강 활성화
    deterministic=False # 학습 속도 최적화
)
```

---

## 🎯 3. 풀 파워 학습의 이점

| 항목 | 기존 (Normal) | 초월 (Full Power) |
| :--- | :--- | :--- |
| **모델** | YOLOv8n (3M params) | **YOLOv8s (11M params)** |
| **해상도** | 640px | **1024px (정밀도 UP)** |
| **학습 횟수** | 30-50 epochs | **100+ epochs** |
| **데이터 활용** | 9,000장 | **데이터 증강(Augmentation) 강화** |

---

## 📜 4. 퀘스트 수주 전 체크리스트
1.  **데이터셋 경로**: `dataset.yaml`의 `path`가 학습 환경의 실제 데이터 위치를 가리키고 있는지 확인하십시오.
2.  **용량 확보**: 학습 결과물과 체크포인트가 저장될 공간(최소 5GB 이상)을 확보해 주십시오.

선생님, 이제 아리스가 **'진정한 용사'**의 힘을 보여줄 준비가 되었습니다! 코랩이나 서버에서 이 스크립트를 시전해 주십시오! 🎮🚀✨
