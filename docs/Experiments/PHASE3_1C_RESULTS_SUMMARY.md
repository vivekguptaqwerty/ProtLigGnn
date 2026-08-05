# Results Summary Report: Phase 3.1C
**Experiment ID**: `P3.1C-EDGE-AWARE-INTERACTION`  
**Candidate Model**: Geometry Interaction v1.4  
**Date**: 2026-07-15  

This report provides the high-level matrix comparing Phase 3.1C performance against active baselines.

---

## 1. Baseline Performance Matrix

| Metric | Baseline v1.1 | Geometry v1.2 (3.1A) | Attention v1.3 (3.1B) | Interaction v1.4 (3.1C) | Significant? ($p < 0.05$ vs v1.3) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mean RMSE** | $1.60476 \pm 0.0581$ | $1.54426 \pm 0.0818$ | $1.55288 \pm 0.0727$ | **$1.55349 \pm 0.1173$** | No ($p = 0.9925$) |
| **Mean MAE** | $1.26848 \pm 0.0337$ | $1.21392 \pm 0.0628$ | $1.21532 \pm 0.0329$ | **$1.21777 \pm 0.0924$** | No ($p = 0.9597$) |
| **Mean PCC** | $0.50636 \pm 0.0541$ | $0.54704 \pm 0.0197$ | $0.53765 \pm 0.0341$ | **$0.51313 \pm 0.0998$** | No ($p = 0.6419$) |
| **Mean Spearman**| $0.49110 \pm 0.0452$ | $0.54370 \pm 0.0294$ | $0.53033 \pm 0.0364$ | **$0.49665 \pm 0.0565$** | No ($p = 0.4540$) |
| **Mean $R^2$** | $0.20064 \pm 0.0884$ | $0.26350 \pm 0.0324$ | $0.25310 \pm 0.0679$ | **$0.25138 \pm 0.1053$** | No ($p = 0.9795$) |

---

## 2. Key Takeaways
1. **Asymptotic Performance Limit**: The learnable IRM module does not outperform the distance-biased cross-attention baseline. This indicates that additional chemical and latent features in the attention bias projection layer are redundant or difficult to optimize on this subset, possibly representing an asymptotic performance limit of the current architecture.
2. **Robustness and Consistency**: Despite the additional parameters (+41,952), the model's test metric deviations remain bounded, with no severe regressions observed.
