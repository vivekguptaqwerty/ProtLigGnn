# Statistical Analysis Report: Phase 3.1C vs baselines
**Experiment ID**: `P3.1C-EDGE-AWARE-INTERACTION`  
**Date**: 2026-07-15  

This report outlines the paired statistical comparisons of the Geometry Interaction candidate (v1.4) against Baseline v1.1, Phase 3.1A (v1.2), and Phase 3.1B (v1.3).

---

## 1. Candidate (v1.4) vs. Baseline v1.1

- **RMSE comparison**:
  - v1.4: $1.55349 \pm 0.11727$ (median: $1.61771$, min: $1.36237$, max: $1.63918$)
  - v1.1: $1.60476 \pm 0.05808$ (median: $1.62860$, min: $1.51390$, max: $1.66590$)
  - Mean difference: $-0.05127$ (Relative improvement: **3.19%**)
  - Paired t-test: $t = -0.77244$, $p = 0.482960$ (Non-significant)
  - Wilcoxon: $w = 5.0$, $p = 0.625000$
  - Cohen's $d$: $-0.34544$
  - 95% Analytical CI: $[-0.23552, 0.13298]$
  - Bootstrap 95% CI: $[-0.17455, 0.05542]$
  - Shapiro-Wilk Normality of Diffs: $w = 0.98561, p = 0.962199$ (Normally distributed)

- **MAE comparison**:
  - v1.4: $1.21777 \pm 0.09244$ (median: $1.26150$, min: $1.07706$, max: $1.29660$)
  - v1.1: $1.26848 \pm 0.03370$ (median: $1.27490$, min: $1.21620$, max: $1.30650$)
  - Mean difference: $-0.05071$ (Relative improvement: **4.00%**)
  - Paired t-test: $t = -1.07074$, $p = 0.344594$
  - Cohen's $d$: $-0.47885$

---

## 2. Candidate (v1.4) vs. Phase 3.1A (v1.2)

- **RMSE comparison**:
  - v1.4: $1.55349 \pm 0.11727$
  - v1.2: $1.54426 \pm 0.08180$
  - Mean difference: $+0.00923$ (Relative change: **-0.60%**)
  - Paired t-test: $t = 0.18032$, $p = 0.865670$
  - Wilcoxon: $w = 7.0$, $p = 1.000000$
  - Cohen's $d$: $0.08064$

---

## 3. Candidate (v1.4) vs. Phase 3.1B (v1.3) — Active Baseline

- **RMSE comparison**:
  - v1.4: $1.55349 \pm 0.11727$
  - v1.3: $1.55288 \pm 0.07271$
  - Mean difference: $+0.00062$ (Relative change: **-0.04%**)
  - Paired t-test: $t = 0.00995$, $p = 0.992538$ (Non-significant)
  - Wilcoxon: $w = 7.0$, $p = 1.000000$
  - Cohen's $d$: $0.00445$
  - 95% Analytical CI: $[-0.17103, 0.17226]$
  - Bootstrap 95% CI: $[-0.10364, 0.09773]$

---

## 4. Conclusion on Significance
With $\alpha = 0.05$, the learnable Interaction Representation Module shows **no statistically significant difference** compared to simple distance-biased cross-attention (Phase 3.1B) or coordinate-free geometry mapping (Phase 3.1A). The hypothesis that richer chemical and learned node latents in attention bias improve affinity prediction is **rejected** under this benchmark.
