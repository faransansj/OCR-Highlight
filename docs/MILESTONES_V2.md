# [QUEST LOG: OCR-Highlight 2.0 Milestone Roadmap]

## 🛡️ Current Status: Level 50 (Mastering the Basics)
아리스는 현재 **Phase 1 (탐지)**을 정복하고 **Phase 2 (다국어 OCR 통합)** 스테이지에 진입했습니다! 선생님이 주신 'Claude Code'라는 전설급 장비 덕분에 퀘스트 진행 속도가 비약적으로 상승 중입니다.

---

## 🗺️ Future Quest Map (The Road to v2.0)

### Milestone 1: [다국어 대현자] (Week 6 - 진행 중)
**"모든 언어를 해석하는 용사의 지혜"**
- **목표**: Tesseract, EasyOCR, PaddleOCR의 힘을 하나로 합치기.
- **핵심 과제**:
    - [x] 3대 OCR 엔진 통합 Wrapper 구현 (`multi_ocr.py`)
    - [x] IoU 기반 결과 병합(Ensemble) 및 투표 시스템 구축
    - [ ] 하이라이트 색상별 맞춤형 정화 마법(Preprocessing) 최적화
    - [ ] 합성 데이터 10,000장 대상 최종 성능 검증 시뮬레이션

### Milestone 2: [차원의 틈새: 현실 세계로] (Week 7-8)
**"가상 세계(합성 데이터)를 넘어 현실의 문서를 정복"**
- **목표**: 실제 스마트폰으로 찍은 문서에서도 95% 이상의 정확도 달성.
- **핵심 과제**:
    - [ ] 실제 데이터 유물(100~200장) 수집 및 각인(Labeling)
    - [ ] **Transfer Learning**: 서버에서 훈련 중인 'Large 모델'을 실제 데이터로 미세 조정
    - [ ] 현실 노이즈(그림자, 번짐)에 강한 Robustness 강화

### Milestone 3: [축소의 마법: 포켓 히어로] (Week 9-10)
**"거대한 지능을 작고 빠른 몸체에 담기"**
- **목표**: Large 모델의 성능을 Nano 모델에 전수(Knowledge Distillation).
- **핵심 과제**:
    - [ ] **Teacher-Student 학습**: Large 모델(Teacher) -> Nano 모델(Student) 지식 전수
    - [ ] 저사양 기기(모바일/임베디드)에서도 '빠밤!' 하고 즉시 실행되도록 경량화 최적화
    - [ ] 추론 속도 0.1초 미만 달성

### Milestone 4: [궁극의 보물: 시스템 알파] (Final)
**"모든 기술이 통합된 완벽한 결과물 소환"**
- **목표**: 탐지부터 OCR까지 한 번에 끝내는 통합 파이프라인 완성.
- **핵심 과제**:
    - [ ] 전 과정 자동화 파이프라인 (`unified_pipeline_v2.py`) 배포
    - [ ] 학술적 성과 정리 (연구 논문 및 최종 리포트 작성)
    - [ ] **보스 레이드 성공!** (프로젝트 최종 완료)

---

## 📊 현재 파티 스탯 (Party Stats)
- **탐지력 (Detection)**: 97% (mAP50, Large Model) ✅
- **해석력 (OCR Accuracy)**: 85% -> 목표 95% 🔄
- **속도 (Inference Speed)**: 0.5s/img ✅
- **사용 가능 마나 (Claude Code)**: 100% (Shared with Sensei) 🔮

선생님, 아리스의 마일스톤이 마음에 드십니까? 아리스는 지금 바로 **'Milestone 1'**의 마지막 조각인 **'맞춤형 정화 마법(Preprocessing)'** 연구에 착수하고 싶습니다! 두둥!
