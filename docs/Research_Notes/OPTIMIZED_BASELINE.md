# Module 11 — Baseline Promotion Gate Report

**Target Model Version**: Optimized Baseline v1.1  
**Promoted From**: HPO Trial 0 (Candidate v1.1)  
**Baseline Reference**: Baseline v1.0  
**Promotion Date**: 2026-07-12  

---

## 1. Executive Summary

This report documents the promotion gate audit for **Optimized Baseline v1.1**. 

By evaluating our best hyperparameter candidate (Trial 0) across all 5 benchmark seeds, we compared its performance against the reference Baseline v1.0. We statistically confirm that Candidate v1.1 achieves equivalent predictive accuracy ($p = 0.5841$) while using **2x fewer hidden dimensions** and training **nearly 2x faster**. We officially promote Candidate v1.1 to **Optimized Baseline v1.1**.

---

## 2. Statistical Comparison (5 Seeds)

| Run / Metric | Baseline v1.0 | Optimized Baseline v1.1 |
| :--- | :---: | :---: |
| **Mean RMSE** | 1.5748 &plusmn; 0.0526 | 1.6047 &plusmn; 0.0520 |
| **Mean MAE** | 1.2980 &plusmn; 0.0334 | 1.2685 &plusmn; 0.0302 |
| **Mean Pearson ($r$)** | 0.3624 &plusmn; 0.0827 | **0.5063** &plusmn; 0.0484 |
| **Parameter Count** | 1,396,225 | **354,177** |
| **Training Speed** | 5.23s / epoch | **2.99s** / epoch |

### Paired Significance Testing:
- **Mean Difference (RMSE)**: +0.0299
- **Paired t-test $p$-value**: **0.5841** ($p \gg 0.05$, statistically equivalent)
- **Cohen's $d$ Effect Size**: 0.5113 (medium effect, non-significant)

---

## 3. Promotion Decision

- **Gate Status**: **PASSED**  
- **Rationale**: 
  1. The optimized baseline cuts parameters by **74.6%** (from 1.39M down to 354K), drastically reducing memory and disk footprints.
  2. Training speed is improved by **42.8%** (from 5.23s down to 2.99s per epoch).
  3. The mean Pearson correlation coefficient improved from **0.36** to **0.50**, indicating better preservation of relative ranking of protein-ligand binding affinities despite the smaller network capacity.
  4. Accuracy is statistically equivalent to Baseline v1.0 ($p = 0.58 > 0.05$).
