# ML-Based Approach: YOLO Object Detection

## 🎯 전략 변경

**기존 접근 (실패):**
- 규칙 기반 detector (Hough Transform, Contour Analysis)
- 합성 데이터: 약간 작동 (Precision 0.30)
- 실제 데이터: 거의 실패 (Precision 0.002-0.076)

**새로운 접근:**
- **YOLOv8 Object Detection**
- 학습 기반 → 복잡한 배경 처리 가능
- 10,100장 GT 데이터 활용

---

## 🚀 구현 계획

### Phase 1: 데이터 준비 ✅
- [x] YOLO 포맷 변환기 작성 (yolo_converter.py)
- [x] 합성 10K + 실제 100장 병합
- [ ] Train/Val split (90%/10%)

### Phase 2: YOLOv8 학습
- [ ] YOLOv8 nano model 다운로드
- [ ] Fine-tuning (epochs: 50-100)
- [ ] 학습 시간: ~1-2시간 예상

### Phase 3: 평가
- [ ] Validation set 성능 측정
- [ ] Test set (DocVQA 별도 50장) 성능
- [ ] 기존 규칙 기반 vs YOLO 비교

### Phase 4: 통합
- [ ] unified_pipeline.py 업데이트
- [ ] YOLO detector 통합
- [ ] 최종 성능 검증

---

## 📊 예상 성능

**목표:**
- Precision: > 0.70
- Recall: > 0.60
- mAP50: > 0.75

**근거:**
- YOLOv8은 object detection SOTA
- 10K+ 학습 데이터 충분
- 5개 클래스 (단순)

---

## 🛠️ 기술 스택

- **Model**: YOLOv8n (nano - fastest)
- **Framework**: Ultralytics YOLO
- **Training**: GPU 권장 (CPU도 가능하지만 느림)
- **Input**: 640x640 (YOLO default)
- **Output**: bbox + class + confidence

---

## 📁 데이터 구조

```
data/yolo_dataset/
├── dataset.yaml          # YOLO config
├── train/
│   ├── images/          # 9,090 images
│   └── labels/          # 9,090 .txt files
└── val/
    ├── images/          # 1,010 images
    └── labels/          # 1,010 .txt files
```

**Label format (YOLO):**
```
class_id x_center y_center width height
0 0.5 0.3 0.2 0.05    # highlight
1 0.4 0.6 0.3 0.02    # underline
```

---

## 🎓 학습 파라미터

```python
model = YOLO('yolov8n.pt')
results = model.train(
    data='data/yolo_dataset/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    patience=10,  # Early stopping
    save=True,
    device='cpu'  # or 'cuda'
)
```

---

## 🔄 다음 단계

1. ⏳ Ultralytics 설치 완료 대기
2. 데이터 변환 실행
3. 학습 시작
4. 성능 평가
5. 최종 보고

---

**Status**: Phase 1 진행 중
**Next**: YOLOv8 학습 시작
