# Phase 3.1B Error Analysis Report
**Experiment ID**: `P3.1B-GEOMETRY-ATTENTION`  
**Status**: Evaluation Completed  
**Date**: 2026-07-15  

---

## 1. Residuals vs. True Affinity Analysis

This report evaluates prediction residuals ($e = \hat{y} - y$) across the combined test predictions (500 samples across the 5 canonical seed runs).

- **Pearson Correlation of Residuals vs True Affinity**: $r = -0.7582$, $p = 1.54 \times 10^{-94}$
- **Linear Regression Slope**: $-0.6497$
- **Linear Regression Intercept**: $4.0771$

### Interpretation
The regression slope of $-0.6497$ represents a minor improvement over the certified Baseline v1.1 slope of $-0.66$. This indicates that while regression-to-the-mean remains a dominant factor in this GNN, the distance-biased cross-attention layer slightly attenuates systematic biases.

---

## 2. Bias by Affinity Range

We segment predictions into three affinity ranges:

| Affinity Range | Count (N) | Mean Residual (P3.1B) | Mean Residual (Baseline v1.1) |
| :--- | :---: | :---: | :---: |
| **Low Affinity (< 4.5)** | 81 | $+1.7934$ | $+1.6400$ |
| **Mid Affinity (4.5 to 7.5)** | 291 | $+0.1392$ | $+0.0400$ |
| **High Affinity (> 7.5)** | 128 | $-1.4763$ | $-1.7200$ |

### Interpretation
The model continues to overpredict weak binders and underpredict strong binders. However, the underprediction of high-affinity complexes ($> 7.5$) improved from $-1.72$ (Baseline) to $-1.48$ (Phase 3.1B). This represents a **14% relative reduction in high-affinity underprediction**.

---

## 3. Best and Worst Predictions

### Best 5 Predictions (Closest to True Value)

| PDB ID | True Affinity | Predicted Affinity | Absolute Error | Seed Run |
| :---: | :---: | :---: | :---: | :---: |
| **3qj9** | 7.0000 | 7.0042 | 0.0042 | Seed 2024 |
| **2y80** | 8.0458 | 8.0501 | 0.0044 | Seed 2024 |
| **3ptg** | 7.6021 | 7.6064 | 0.0044 | Seed 3407 |
| **3pgu** | 6.6990 | 6.6940 | 0.0050 | Seed 2024 |
| **3arf** | 7.1938 | 7.2141 | 0.0202 | Seed 3407 |

### Worst 5 Predictions (Largest Absolute Error)

| PDB ID | True Affinity | Predicted Affinity | Absolute Error | Seed Run |
| :---: | :---: | :---: | :---: | :---: |
| **2xpk** | 11.3010 | 6.0727 | 5.2283 | Seed 42 |
| **3ao4** | 2.0706 | 7.0377 | 4.9671 | Seed 2024 |
| **2xpk** | 11.3010 | 6.4255 | 4.8755 | Seed 2024 |
| **2xiy** | 2.4437 | 6.8775 | 4.4338 | Seed 42 |
| **3p7i** | 7.7011 | 3.2792 | 4.4219 | Seed 2024 |

The worst predicted complex is `2xpk`, an extremely strong binder ($pK_d = 11.3$) which is heavily underpredicted. This aligns with the regression-to-the-mean limit, where the model compresses extreme outliers toward the training mean ($\sim 6.3$).
