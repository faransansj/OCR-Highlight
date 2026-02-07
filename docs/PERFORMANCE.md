# 📊 Performance Metrics

Detailed performance evaluation of the Highlight Detection and OCR components.

---

## 🎯 YOLO Markup Detection (Current)

Results from **Epoch 1** (Early checkpoint).

| Metric | Value |
| :--- | :--- |
| **mAP50** | **96.10%** |
| mAP50-95 | 75.44% |
| Precision | 98.80% |
| Recall | 63.31% |

**Interpretation:**
The model is extremely precise (low false positives) but conservative. High confidence in its detections.

---

## 🏛️ Legacy HSV Pipeline

### Highlight Detection (mIoU)
| Metric | Value | Status |
| :--- | :--- | :--- |
| **mIoU** | **0.8222** | ✅ Passed (>0.75) |
| Precision | 0.7660 | ✅ |
| Recall | 0.5806 | ✅ |

### OCR Accuracy
| Metric | Value | Target |
| :--- | :--- | :--- |
| **Accuracy** | **95.30%** | >95% |
| **CER** | **4.70%** | <5% |

---

## 📈 Improvement History

| Stage | CER | Accuracy | Key Optimization |
| :--- | :--- | :--- | :--- |
| Initial | 80.95% | 19.05% | - |
| Lang Fix | 46.83% | 53.17% | `kor+eng` support |
| Space Fix | 24.83% | 75.17% | Korean space removal |
| Multi-PSM | 8.39% | 91.61% | Multi-PSM + de-duplication |
| **Final** | **4.70%** | **95.30%** | **Particle restoration** |

---

[⬅ Back to README](../README.md)
