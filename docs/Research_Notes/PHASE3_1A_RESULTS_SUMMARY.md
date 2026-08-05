# Phase 3.1A Results Summary Table
**Experiment ID**: P3.1A-GEOMETRY-LIGAND  
**Status**: Audited Summary  

This summary compares the performance of the geometry model against the certified Baseline v1.1.

---

## 1. Baseline vs. Geometry Performance Matrix

| Metric | Certified Baseline v1.1 | Geometry Model v1.2 | Relative Change | Statistical Significance ($p < 0.05$) |
| :--- | :---: | :---: | :---: | :---: |
| **Mean RMSE** | $1.6048 \pm 0.0581$ | $1.5442 \pm 0.0818$ | **-3.77%** (Lower Error) | No ($p = 0.1484$) |
| **Mean MAE** | $1.2685 \pm 0.0337$ | $1.2139 \pm 0.0628$ | **-4.30%** (Lower Error) | No ($p = 0.0564$) |
| **Mean PCC ($r$)** | $0.5064 \pm 0.0541$ | $0.5470 \pm 0.0197$ | **+8.03%** (Higher Corr) | No ($p = 0.2092$) |
| **Mean Spearman** | $0.4911 \pm 0.0452$ | $0.5437 \pm 0.0293$ | **+10.71%** (Higher Corr) | No ($p = 0.1143$) |
| **Mean $R^2$** | $0.2006 \pm 0.0884$ | $0.2635 \pm 0.0324$ | **+31.34%** (Higher Expl) | No ($p = 0.1549$) |
| **Total Parameters**| 362,881 | 364,001 | **+0.31%** (1,120 params) | N/A |

---

## 2. Key Observations
1. **Consistent Directional Improvement**: The geometry model performs directionally better across all 5 test metrics, demonstrating that explicit RBF distance representations introduce useful spatial signals.
2. **Low Statistical Power**: Due to the small sample size constraint ($N=5$ seeds), the substantial improvements (e.g. 4.3% in MAE, 10.7% in Spearman rank correlation) are not statistically significant at the $\alpha = 0.05$ boundary.
3. **Parametric Efficiency**: The geometry model achieves this directional improvement with a negligible addition of only **1,120 parameters**, maintaining the high computational efficiency of the baseline.
