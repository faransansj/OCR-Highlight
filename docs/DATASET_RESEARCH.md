# Document Markup Dataset Research

## 🔍 검색 필요 (Brave API 키 없음)

웹 검색 실패 - API 키 설정 필요:
```bash
openclaw configure --section web
```

---

## 📚 알려진 관련 데이터셋 (일반 지식)

### 1. **DocBank** (Microsoft)
- **용도**: Document layout analysis
- **내용**: 50만+ 문서 페이지, bbox annotations
- **마크업**: ❌ (레이아웃만, 하이라이트/밑줄 없음)
- **링크**: GitHub microsoft/DocBank

### 2. **PubLayNet** (IBM)
- **용도**: Document layout detection
- **내용**: 36만+ 학술 논문 페이지
- **마크업**: ❌ (레이아웃만)
- **링크**: GitHub ibm-aur-nlp/PubLayNet

### 3. **IAM Handwriting Database**
- **용도**: Handwritten text recognition
- **마크업**: ⚠️ (일부 annotation 있을 수 있음)
- **링크**: fki.tic.heia-fr.ch

### 4. **FUNSD** (Form Understanding)
- **용도**: Form understanding, entity linking
- **마크업**: ❌ (폼 필드만)
- **링크**: guillaumejaume/FUNSD

### 5. **ICDAR Competition Datasets**
- **시리즈**: ICDAR 2013-2023
- **관련 트랙**: Document analysis, layout analysis
- **마크업**: ⚠️ (특정 연도에 annotation task 있을 수 있음)
- **링크**: ICDAR 공식 사이트

---

## 🎯 우리 프로젝트에 필요한 것

**필요한 데이터:**
- 하이라이트된 텍스트
- 밑줄/취소선
- 동그라미/네모 표시
- Ground truth bbox + type

**현재 상황:**
- 직접 매칭되는 공개 데이터셋은 **거의 없음** (일반 지식 기준)
- 대부분 layout detection용 (마크업 감지용 아님)

---

## 💡 추천 전략

### Option 1: 유사 데이터셋 활용
1. **DocBank / PubLayNet** 다운로드
2. 문서 이미지에 **우리가 직접 마크업 추가**
3. 반자동 annotation (우리 synthetic generator 활용)

### Option 2: 커뮤니티 검색
- **Kaggle Datasets**: "document annotation", "highlight detection"
- **Papers With Code**: Datasets 섹션 검색
- **Hugging Face Datasets Hub**: `datasets` 라이브러리
- **Google Dataset Search**: dataset-specific 검색

### Option 3: 크라우드소싱 최소화
- 100장만 수동 수집 (우리가 직접 책 사진)
- 나머지는 synthetic + transfer learning

---

## 🚀 즉시 실행 가능한 작업

### 1. Hugging Face 검색 (API 키 불필요)
```python
from datasets import load_dataset_builder
# Search for document-related datasets
```

### 2. Kaggle 수동 검색
- "document markup"
- "highlighted text"
- "annotation detection"

### 3. Papers With Code
- Browse datasets by task
- "Document Understanding" 카테고리

---

## ❓ 다음 단계

**선생님께 질문:**
1. Brave API 키 설정해주실 수 있나요? → 자동 검색 가능
2. 아니면 Alice가 Hugging Face/Kaggle을 수동으로 탐색할까요?
3. 또는 100장 직접 수집(우리가 책 촬영) 후 synthetic 활용?

**추천**: Option 2 → 웹 검색 없이 Hugging Face Datasets API로 탐색

---

**Status**: Research paused - need direction or API key
**Next**: Await user input for search strategy
