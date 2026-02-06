# Performance Improvement Log

## Iteration 1: Filter Optimization (2026-02-06)

**Problem:** High False Positive rate (Precision 10-12%)

**Hypothesis:** Detectors too sensitive, picking up text/noise as markups

**Changes:**
1. Line Detector:
   - `min_line_length`: 20 → 30 pixels
   - Added `min_confidence` filter: 0.5
   
2. Shape Detector:
   - `min_area`: 100 → 500 pixels
   - `min_confidence`: 0.5 → 0.65

**Results:**
| Detector | Metric | Before | After | Target | Status |
|----------|--------|--------|-------|--------|--------|
| Line | Precision | 0.1035 | 0.1137 | > 0.70 | ❌ |
| Line | Recall | 0.9119 | 1.0000 | > 0.60 | ✅ |
| Line | F1 | 0.1859 | 0.2043 | > 0.65 | ❌ |
| Shape | Precision | 0.1186 | 0.3014 | > 0.65 | ❌ |
| Shape | Recall | 0.3236 | 0.3099 | > 0.55 | ❌ |
| Shape | F1 | 0.1736 | 0.3056 | > 0.60 | ❌ |

**Analysis:**
- ✅ Shape detector showed significant improvement (Precision +154%, F1 +76%)
- ⚠️ Still below production targets
- 🔍 Root cause: **Synthetic data quality issue**
  - Markups rendered too similarly to text
  - Need clearer visual distinction
  - Consider: thicker lines, larger shapes, more contrast

**Next Steps:**
1. Improve synthetic data generator:
   - Make markups more visually distinct
   - Add more variation in rendering
   - Better separation from text
2. Alternative: Switch focus to real data collection earlier
3. Consider: Implement machine learning classifier instead of rule-based

**Status:** Partial improvement, need different approach

---

## Next Iteration Plan

**Strategy A (Quick Win):** Improve synthetic data quality
- Render markups with more visual distinction
- Increase size/thickness constraints
- Better color/style variation

**Strategy B (Long-term):** Accelerate real data collection
- Skip to Week 8 plan early
- Collect 1,000 real samples
- Train on real data

**Recommendation:** Try Strategy A first (1-2 days), then Strategy B if needed

