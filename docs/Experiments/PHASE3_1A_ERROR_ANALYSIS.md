# Phase 3.1A Detailed Error Analysis Report
**Experiment ID**: P3.1A-GEOMETRY-LIGAND  
**Model**: ProtLigGNNGeometry (v1.2)  
**Baseline Reference**: Certified Optimized Baseline v1.1  
**Status**: Completed  

This report provides a systematic analysis of prediction errors, residuals, and calibration of the geometry model compared to the baseline.

---

## 1. Quantitative Residual & Trend Analysis

### 1.1 Regression-to-the-Mean Trendline
We observe that the regression-to-the-mean bias remains practically unchanged:
- **Baseline Correlation ($r$)**: $-0.75$
- **Geometry Correlation ($r$)**: $-0.7722$
- **Baseline Trendline Slope**: $-0.66$
- **Geometry Trendline Slope**: $-0.6542$

*Interpretation*: Hypothesis $H_2$ (explicit geometry reduces regression-to-the-mean) is **rejected**. The model still exhibits a strong regression-to-the-mean bias, collapsing predictions toward the dataset mean ($\approx 6.0$).

### 1.2 Bias by Affinity Range
Grouping predictions by true binding affinity values:
- **Low-Affinity Group (True Affinity $< 4.5$)**: Mean residual of **$+1.7205$** (Baseline was $+1.64$). Systematic over-prediction is slightly worsened.
- **High-Affinity Group (True Affinity $> 7.5$)**: Mean residual of **$-1.6584$** (Baseline was $-1.72$). Systematic under-prediction is marginally mitigated.

---

## 2. Baseline Outlier Predictions in Geometry Model

We analyze the predictions of the top 10 highest-error complexes identified in the Baseline v1.1 audit:

| PDB ID | True Affinity | Baseline Pred Mean | Geometry Pred Mean | Baseline Residual | Geometry Residual | Performance Delta |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **2xpk** | 11.30 | 6.43 | 6.46 | -4.87 | -4.85 | Equivalent |
| **3rlb** | 9.92 | 5.20 | 6.43 | -4.73 | -3.50 | **Significant Improvement (+1.23)** |
| **2xn7** | 10.00 | 5.35 | 5.28 | -4.65 | -4.72 | Marginal Regression (-0.07) |
| **2xiy** | 2.44 | 6.85 | 5.66 | +4.41 | +3.22 | **Significant Improvement (+1.19)** |
| **3p7i** | 7.70 | 3.31 | 3.63 | -4.39 | -4.07 | Marginal Improvement (+0.32) |
| **3rbu** | 9.43 | 5.24 | 5.83 | -4.19 | -3.60 | **Improvement (+0.59)** |
| **2yki** | 9.46 | 5.30 | 5.35 | -4.16 | -4.11 | Equivalent |
| **3ao4** | 2.07 | 6.20 | 6.44 | +4.13 | +4.37 | Regression (-0.24) |
| **2ll6** | 10.00 | 5.89 | 6.20 | -4.11 | -3.80 | **Improvement (+0.31)** |
| **2yiv** | 2.78 | 6.70 | 6.00 | +3.92 | +3.22 | **Improvement (+0.70)** |

*Observation*: Explicit geometry significantly resolves errors for some severe outliers (such as `3rlb` and `2xiy`), indicating that physical bond length details help the model identify distinct structure-property limits. However, the model still collapses predictions for ultra-high affinity binders (e.g. `2xpk`), which is primarily a limitation of training set sample distributions.
