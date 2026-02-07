# 데이터셋 검색 결과 종합

## 🔍 검색 완료 (2026-02-07)

### Hugging Face Datasets 검색

**검색어별 결과:**
- "document annotation": 0개
- "highlight": 68개 (대부분 비디오/텍스트 하이라이트)
- "markup": 8개 (HTML/XML 마크업 관련)
- "document understanding": 1개

---

## ⭐ 유망한 후보

### 1. **ademax/ocr_highlight_vi** 🔒
- **URL**: https://huggingface.co/datasets/ademax/ocr_highlight_vi
- **크기**: 174k rows
- **내용**: OCR + highlight (베트남어)
- **상태**: ⚠️ **로그인 및 약관 동의 필요**
- **우선순위**: **HIGH** - 가장 관련성 높음!

### 2. knkarthick/highlightsum
- **URL**: https://huggingface.co/datasets/knkarthick/highlightsum
- **크기**: 31.1k rows
- **내용**: Text summarization with highlights
- **우선순위**: MEDIUM - 텍스트 하이라이트 요약용

### 3. MarkupLM 관련 데이터셋
- `nielsr/markuplm-toy-dataset`
- `ryo0634/squad-with-markup`
- **내용**: HTML/XML markup (문서 마크업 아님)
- **우선순위**: LOW

---

## 🚫 제한사항

### Kaggle
- reCAPTCHA로 자동 검색 차단
- 수동 접근 필요

### Papers With Code
- Redirect → Hugging Face Papers (데이터셋 직접 검색 불가)

---

## 💡 추천 전략

### Option A: ademax/ocr_highlight_vi 확인
**장점:**
- 가장 관련성 높음 (OCR + highlight)
- 174k rows = 충분한 규모

**단점:**
- 로그인 필요 (승준님이 직접 해야 함)
- 베트남어 중심 (한/영/일/중과 다름)
- 약관 확인 필요 (라이선스, 사용 제한)

**다음 단계:**
1. 승준님이 Hugging Face 로그인
2. 데이터셋 약관 확인
3. 샘플 다운로드 후 형식 분석
4. 우리 프로젝트에 적합한지 검증

### Option B: 기존 계획 유지
- 100장 직접 수집 (책 촬영)
- 10,000장 합성 데이터 활용
- Transfer learning

### Option C: 하이브리드
- ademax 데이터셋 사용 (베트남어)
- + 우리가 100장 추가 (한/영/일/중)
- = 다국어 robust model

---

## 📋 다음 작업

**즉시 가능:**
1. ✅ 검색 완료
2. ⏳ 승준님이 ademax 데이터셋 확인

**대기 중:**
1. Brave API 키 설정 시 더 광범위한 웹 검색
2. 브라우저 설정 시 Kaggle 수동 탐색
3. 직접 데이터 수집 시작

---

## 🎯 결론

**가장 유망한 데이터셋 발견:**
`ademax/ocr_highlight_vi` (174k, OCR+highlight)

**하지만:**
- 로그인 필요
- 라이선스 확인 필요
- 베트남어 중심

**추천:**
승준님이 해당 데이터셋 확인 후 사용 가능 여부 결정!

---

**Status**: 검색 완료, 유망 후보 1개 발견
**Next**: 승준님의 ademax 데이터셋 접근 및 검증
