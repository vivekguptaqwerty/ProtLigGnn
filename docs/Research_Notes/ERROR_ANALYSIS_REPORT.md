# Error Analysis & Model Calibration Report
## ProtLigGNN Optimized Baseline v1.1

**Analysis Scope**: Test Set predictions aggregated across all **5 official seeds** ($5 \times 100 = 500$ evaluations)  
**Aggregate MAE**: 1.27  
**Aggregate RMSE**: 1.60  
**Analysis Date**: 2026-07-12  

---

## 1. Executive Summary

This report provides a systematic analysis of prediction errors, residuals, and calibration of **ProtLigGNN Optimized Baseline v1.1**. Using the quad-panel diagnostic plots, we reveal a severe regression-to-the-mean bias, identify structural features that correlate with large prediction errors, and compile an outlier catalogue of high-error complexes.

---

## 2. Error Diagnostics Panel

The figure below displays the residual distribution, probplot (Q-Q), calibration curve, and regression-to-the-mean trends:

![Error Diagnostics Panel](file:///C:/Users/Harshhhh/.gemini/antigravity/brain/03c21182-9e56-4764-a6fd-83660d6d0fc1/error_analysis_quad_panel.png)

---

## 3. Quantitative Residual & Calibration Analysis

### 3.1 Normal Q-Q & Residual Distribution
- The residuals (True - Predicted) fit a normal distribution with mean $\mu \approx -0.04$ and standard deviation $\sigma \approx 1.59$.
- The Q-Q plot confirms that residuals are approximately normal between $-2.0$ and $+2.0$ standard deviations, but show heavier tails (extreme outliers) at the boundaries.

### 3.2 Regression-to-the-Mean Phenomenon
We observe a strong negative correlation between the true affinity and the residual values:
- **Correlation ($r$)**: $-0.75$ ($p < 10^{-5}$)
- **Slope of Trendline**: $-0.66$
- **High-Affinity Group (True Affinity $> 7.5$)**: Mean residual of **$-1.72$** (systematic under-prediction).
- **Low-Affinity Group (True Affinity $< 4.5$)**: Mean residual of **$+1.64$** (systematic over-prediction).

*Implication: The GNN behaves conservatively, collapsing predictions toward the dataset mean ($\approx 6.0$). This represents a major bottleneck for virtual screening, as it leads to high false-positive rates for low-affinity candidates and false-negative rates for lead-like high-affinity candidates.*

---

## 4. Outlier Catalogue (Top 10 Absolute Errors)

The table below lists the top 10 test set complexes ranked by absolute mean error:

| PDB ID | True Affinity | Predicted Mean | Mean Residual | Absolute Error | Ligand Size (Atoms) | Pocket Size (Residues) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2xpk** | 11.30 | 6.43 | -4.87 | 4.87 | 53 | 36 |
| **3rlb** | 9.92 | 5.20 | -4.73 | 4.73 | 35 | 34 |
| **2xn7** | 10.00 | 5.35 | -4.65 | 4.65 | 71 | 30 |
| **2xiy** | 2.44 | 6.85 | +4.41 | 4.41 | 20 | 22 |
| **3p7i** | 7.70 | 3.31 | -4.39 | 4.39 | 15 | 21 |
| **3rbu** | 9.43 | 5.24 | -4.19 | 4.19 | 24 | 31 |
| **2yki** | 9.46 | 5.30 | -4.16 | 4.16 | 52 | 37 |
| **3ao4** | 2.07 | 6.20 | +4.13 | 4.13 | 27 | 27 |
| **2ll6** | 10.00 | 5.89 | -4.11 | 4.11 | 302 | 57 |
| **2yiv** | 2.78 | 6.70 | +3.92 | 3.92 | 20 | 23 |

---

## 5. Structural Hypotheses for Model Failures

1. **Macromolecular Ligand Scaling Outliers**: Complex `2ll6` has a massive ligand of **302 heavy atoms** (average ligand size in dataset is $\approx 28$ atoms). The GNN utilizes a global mean pooling readout which averages node representations, heavily diluting the local binding coordinate signals in large molecules.
2. **Extreme Affinity Under-representation**: The worst predicted complex, `2xpk` (True affinity 11.30, predicted 6.43, error 4.87), represents an ultra-high affinity binder. Because the training set lacks sufficient examples above 10.0, the L1 loss function fails to penalize large errors at the tails sufficiently to overcome the dataset mean regularization.
