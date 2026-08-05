# Phase 3.1C Statistical Analysis Report
**Experiment ID**: `P3.1C-INTERACTION-REPRESENTATION`  
**Date**: 2026-07-16  

This report provides the paired statistical comparison tests between Candidate v1.4 and reference baselines.

---

## 1. Candidate v1.4 vs. Baseline v1.3 (Attention)

- **Mean Difference**: $+0.00062$ RMSE (Slight increase in mean error)
- **Relative Difference**: **-0.04%**
- **Paired t-test**: \(t = 0.00995\), \(p = 0.992538\) (Non-significant)
- **Wilcoxon Signed-Rank Test**: \(w = 7.0\), \(p = 1.000000\)
- **Cohen's $d$**: \(0.00445\) (Negligible effect)
- **Bootstrap 95% Confidence Interval**: \([-0.10364, 0.09773]\) (Contains zero)
- **Shapiro-Wilk Normality of Diffs**: \(w = 0.97232\), \(p = 0.889921\) (Normality holds)

---

## 2. Candidate v1.4 vs. Geometry v1.2

- **Mean Difference**: $+0.00923$
- **Relative Difference**: **-0.60%**
- **Paired t-test**: \(t = 0.18032\), \(p = 0.865670\)

---

## 3. Candidate v1.4 vs. Optimized Baseline v1.1

- **Mean Difference**: $-0.05127$
- **Relative Difference**: **+3.19%**
- **Paired t-test**: \(t = -0.77244\), \(p = 0.482960\)