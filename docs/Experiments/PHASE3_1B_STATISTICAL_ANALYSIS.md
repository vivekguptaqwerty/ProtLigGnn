# Phase 3.1B Statistical Analysis Report
**Experiment ID**: `P3.1B-GEOMETRY-ATTENTION`  
**Status**: Evaluation Completed  
**Date**: 2026-07-15  

---

## 1. Experimental Overview & Per-Seed Metrics

This report evaluates the performance of the **Distance-Biased Cross-Attention** architecture against the certified `Optimized Baseline v1.1` and `Phase 3.1A (Geometry)`.

### Per-Seed Metric Summary

| Model | Seed 42 | Seed 123 | Seed 777 | Seed 2024 | Seed 3407 | Mean $\pm$ Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline v1.1 (RMSE)** | 1.5863 | 1.6286 | 1.5139 | 1.6659 | 1.6291 | **1.6048 $\pm$ 0.0520** |
| **Phase 3.1A (RMSE)** | 1.4616 | 1.5727 | 1.5389 | 1.6671 | 1.4810 | **1.5442 $\pm$ 0.0818** |
| **Phase 3.1B (RMSE)** | 1.5372 | 1.5564 | 1.4588 | 1.6626 | 1.5493 | **1.5529 $\pm$ 0.0727** |
| | | | | | | |
| **Baseline v1.1 (MAE)** | 1.2162 | 1.2603 | 1.2749 | 1.3065 | 1.2845 | **1.2685 $\pm$ 0.0301** |
| **Phase 3.1A (MAE)** | 1.1502 | 1.2162 | 1.2429 | 1.3021 | 1.1582 | **1.2139 $\pm$ 0.0628** |
| **Phase 3.1B (MAE)** | 1.1928 | 1.1982 | 1.1919 | 1.2691 | 1.2247 | **1.2153 $\pm$ 0.0329** |

---

## 2. Paired Hypothesis Testing (P3.1B vs Baseline v1.1)

We perform a paired two-sample t-test across the 5 canonical seeds:

### 2.1 Root Mean Squared Error (RMSE)
- **Mean Difference**: $-0.0519$ (RMSE reduction)
- **Paired t-Test**: $t = -3.8841$, **$p = 0.01778$**
- **Wilcoxon Signed-Rank**: $w = 0.00$, $p = 0.0625$
- **Cohen's d**: $-1.7370$ (Very large effect size)
- **95% Bootstrap CI of Diff**: $[-0.0721, -0.0274]$
- **Shapiro-Wilk Normality**: $W = 0.8916$, $p = 0.3655$ (Differences are normally distributed; t-test is valid)

### 2.2 Mean Absolute Error (MAE)
- **Mean Difference**: $-0.0532$ (MAE reduction)
- **Paired t-Test**: $t = -5.1227$, **$p = 0.00687$**
- **Cohen's d**: $-2.2909$ (Extremely large effect size)
- **95% Bootstrap CI of Diff**: $[-0.0700, -0.0353]$

### 2.3 Coefficient of Determination ($R^2$)
- **Mean Difference**: $+0.0525$ ($R^2$ improvement)
- **Paired t-Test**: $t = 3.6797$, **$p = 0.02121$**

---

## 3. Direct Comparison (P3.1B vs Phase 3.1A)

We perform paired tests comparing the two geometry-aware implementations:

- **RMSE Mean Difference**: $+0.0086$ (Phase 3.1A is slightly lower in mean, but not statistically significant)
- **Paired t-Test (RMSE)**: $t = 0.2980$, **$p = 0.7805$**
- **MAE Mean Difference**: $+0.0014$
- **Paired t-Test (MAE)**: $t = 0.0620$, **$p = 0.9535$**

### Scientific Interpretation
Phase 3.1B yields statistically equivalent performance to Phase 3.1A. However, Phase 3.1B achieves this using a coordinate-free GAT encoder identical to the baseline, introducing only **133 extra parameters** (vs 13,728 extra parameters in Phase 3.1A).
