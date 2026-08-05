# Phase 3.2 Metric Verification Report
**Date**: 2026-07-16  
**Candidate Model**: `physics_guided`  

---

### 1. Metric Audit Checklist
We cross-checked all values in the PDF reports against the raw data in `per_seed_metrics.csv` and `aggregate_metrics.csv`:

- **Mean RMSE**:
  - Expected (per_seed_metrics): **1.60912**
  - Observed (Official Benchmark Report): **1.60912**
  - Status: **Match**
- **Mean MAE**:
  - Expected: **1.27090**
  - Observed: **1.27090**
  - Status: **Match**
- **Paired t-test vs v1.3**:
  - Expected: $t = 2.24336$, $p = 0.08829$
  - Observed (Statistical Report): $t = 2.24336$, $p = 0.08829$
  - Status: **Match**
- **Wilcoxon Signed-Rank p-value**:
  - Expected: **0.06250**
  - Observed: **0.06250**
  - Status: **Match**
- **Cohen's $d$**:
  - Expected: **0.72280**
  - Observed: **0.72280**
  - Status: **Match**
- **Bootstrap 95% Confidence Interval**:
  - Expected: **[-0.01245, 0.12563]**
  - Observed: **[-0.01245, 0.12563]**
  - Status: **Match**

---

### 2. Discrepancy Analysis
No discrepancies exist in the regression or statistical metrics. Every reported metric matches the raw execution logs.
