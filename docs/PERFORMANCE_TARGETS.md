# Performance Targets - OCR-Highlight 2.0

## 📊 Month 1-2 Benchmark Goals

### 1. Highlight Detection (기존 기능 유지/향상)
**Target Performance:**
- mIoU (Mean Intersection over Union): **> 0.82** (기존 0.8222 유지)
- Precision: **> 0.75** (기존 0.7660 유지)
- Recall: **> 0.60** (기존 0.5806 향상)
- F1-Score: **> 0.65** (기존 0.6606 유지)

**Validation Set:** 150 images with yellow/green/pink highlights

---

### 2. Line Detection (NEW - Underlines & Strikethroughs)
**Target Performance:**
- mIoU: **> 0.75** (새로운 기능)
- Precision: **> 0.80** (라인은 명확하므로 높은 precision 목표)
- Recall: **> 0.70** 
- F1-Score: **> 0.70**

**Specific Metrics:**
- Underline Detection Accuracy: **> 85%**
- Strikethrough Detection Accuracy: **> 85%**
- Double-line Detection Rate: **> 80%** (겹침 감지)
- False Positive Rate: **< 10%** (텍스트 밑선과 혼동 최소화)

**Validation Set:** 200 images
- 100 with underlines (50 single, 50 double)
- 100 with strikethroughs (50 single, 50 double)

---

### 3. Shape Detection (NEW - Circles, Rectangles, etc.)
**Target Performance:**
- mIoU: **> 0.70** (도형은 다양하므로 slightly lower)
- Precision: **> 0.75**
- Recall: **> 0.65**
- F1-Score: **> 0.65**

**Per-Shape Accuracy:**
- Circle Detection: **> 85%** (circularity가 명확)
- Rectangle Detection: **> 80%**
- Star Detection: **> 70%** (복잡한 패턴)
- Checkmark Detection: **> 75%**

**Validation Set:** 200 images
- 50 circles
- 50 rectangles  
- 50 stars
- 50 checkmarks

---

### 4. Unified Pipeline (Integration)
**Target Performance:**
- Overall mIoU: **> 0.75** (모든 마크업 통합)
- Multi-markup Detection: **> 80%** (한 이미지에 여러 마크업)
- Overlap Resolution Accuracy: **> 90%** (우선순위 처리)
- Processing Speed: **< 1.0 sec/image** (800x600 기준)

**Validation Set:** 300 images
- 100 single markup
- 100 dual markups (2 types)
- 100 triple+ markups (3+ types)

---

### 5. Synthetic Data Quality
**Target Metrics:**
- Ground Truth Accuracy: **100%** (자동 생성이므로 완벽)
- Visual Realism Score: **> 0.7** (수동 평가)
- Language Coverage: **4+ languages**
- Markup Diversity: **10+ types**
- Generation Speed: **> 100 images/minute**

**Test:** Generate 1,000 samples and validate

---

## 🔬 Test Execution Plan

### Phase 1: Module Unit Tests (Today)
1. **Line Detector Test**
   - Generate 200 synthetic line samples
   - Measure precision, recall, F1
   - Target: All metrics > 75%

2. **Shape Detector Test**
   - Generate 200 synthetic shape samples
   - Measure per-shape accuracy
   - Target: Circle/Rectangle > 80%

3. **Synthetic Generator Test**
   - Generate 1,000 mixed samples
   - Validate GT correctness
   - Measure generation speed

### Phase 2: Integration Test (Today)
4. **Unified Pipeline Test**
   - Process 300 mixed samples
   - Measure end-to-end performance
   - Validate overlap resolution

### Phase 3: Performance Analysis (Today)
5. **Error Analysis**
   - Identify failure cases
   - Categorize error types
   - Prioritize improvements

---

## ✅ Success Criteria

**Minimum Viable Performance (MVP):**
- Highlights: mIoU > 0.80 ✅
- Lines: mIoU > 0.70 ✅
- Shapes: mIoU > 0.65 ✅
- Pipeline: Speed < 1.5 sec/image ✅

**Target Performance (Production-Ready):**
- All detectors: mIoU > 0.75
- Processing: < 1 sec/image
- Multi-markup: > 80% accuracy

**Stretch Goals:**
- All detectors: mIoU > 0.80
- Real-time processing: < 0.5 sec/image
- Multi-markup: > 90% accuracy

---

**Test Start Time:** 2026-02-06 17:45 UTC
**Expected Duration:** 1-2 hours
**Output:** Performance report + failure analysis
