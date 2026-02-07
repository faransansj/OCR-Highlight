# Real Data Collection Plan

## Background

**Synthetic data limitations discovered:**
- Shape detector: Max precision 0.38 (target: 0.65)
- Line detector: Max precision 0.05 (target: 0.70)
- Root cause: Synthetic markups too similar to text/noise

**Decision:** Pivot to real Ground Truth data collection

---

## Collection Strategy

### Phase 1: Quick Prototype (Week 1)
**Goal:** 100 real samples for proof-of-concept

**Sources:**
1. **Internal collection** (50 samples)
   - Team members scan their own study materials
   - Textbooks, notes with actual highlights/underlines
   - Manual annotation using labeling tool

2. **Public datasets** (50 samples)
   - Academic papers with annotations (arXiv)
   - Open educational resources
   - Creative Commons documents

**Annotation format:**
- JSON matching current synthetic data format
- Tools: LabelImg, CVAT, or custom annotation tool

---

### Phase 2: Scaled Collection (Week 2-3)
**Goal:** 1,000-5,000 samples

**Method A: Crowdsourcing**
- Platform: Amazon MTurk / 크몽 / 자체 플랫폼
- Task: Upload photos of highlighted textbooks/notes
- Payment: $0.50-1.00 per valid sample
- Quality control: Manual review + auto-validation

**Method B: Partnerships**
- Target: Universities, study cafes, educational publishers
- Offer: Free OCR tool in exchange for anonymized data
- Privacy: Remove personal info, focus on markup patterns

**Method C: Synthetic-to-Real Refinement**
- Use synthetic data to pre-train
- Fine-tune on small real dataset (transfer learning)
- Iterate based on real-world performance

---

### Phase 3: Validation (Week 4)
**Goal:** Achieve target performance on real data

**Test split:**
- Train: 70% (700-3,500 samples)
- Validation: 15% (150-750 samples)
- Test: 15% (150-750 samples)

**Success criteria:**
- Line detector: Precision >0.70, Recall >0.60
- Shape detector: Precision >0.65, Recall >0.55
- End-to-end pipeline: Speed <1 sec/image

---

## Immediate Next Steps (This Week)

### 1. Build Annotation Tool (Day 1-2)
Simple web-based tool for labeling:
- Upload image
- Draw bboxes for markups
- Classify type (underline/circle/etc.)
- Export JSON

### 2. Collect Initial 100 Samples (Day 3-5)
- 20 samples: Personal materials
- 30 samples: arXiv papers
- 50 samples: Web scraping (CC-licensed)

### 3. Baseline Test (Day 6-7)
- Test current detectors on real data
- Measure actual performance gap
- Adjust strategy based on results

---

## Budget Estimate

| Item | Cost | Notes |
|------|------|-------|
| Crowdsourcing (1,000 samples @ $0.75) | $750 | Conservative estimate |
| Annotation tool development | $0 | Build in-house |
| Storage (cloud) | $50 | For dataset hosting |
| **Total** | **$800** | Down from $5,000 original plan |

---

## Alternative: Quick Win Approach

If time/budget constrained, try **Method C first:**

1. Generate 10,000 high-quality synthetic samples
2. Train detector on synthetic
3. Collect only 100-200 real samples for fine-tuning
4. Use domain adaptation techniques

**Pros:**
- Faster (1 week vs 4 weeks)
- Cheaper ($0 vs $800)
- Leverages existing synthetic pipeline

**Cons:**
- May still hit accuracy limits
- Real data eventually needed for production

---

## Decision Point

**Recommended:** Try Quick Win Approach first
- If works: Save time & money
- If fails: Full real data collection as backup

**Timeline:**
- Quick Win: 1 week
- Full collection: 4 weeks
- Total max: 5 weeks

---

**Status:** Plan complete, ready for execution
**Next:** Build annotation tool OR try Quick Win approach
