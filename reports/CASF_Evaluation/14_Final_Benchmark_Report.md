# 14 Final Benchmark Report: CASF-2016 Validation

This document presents the consolidated external validation results for the frozen **LigProtGNN-X** model v4.0 on the CASF-2016 benchmark.

## Performance Overview

### 1. Scoring Power (pK$_d$/pK$_i$ Prediction)
| Metric | Value |
| :--- | :--- |
| **Pearson Correlation (R)** | 0.371 (95% CI: [0.275, 0.461]) |
| **Standard Deviation of Fit (SD)** | 2.02 |
| **Spearman Correlation (SP)** | 0.385 |
| **Kendall's Tau (tau)** | 0.257 |
| **Root Mean Squared Error (RMSE)** | 2.111 (95% CI: [1.950, 2.266]) |
| **Mean Absolute Error (MAE)** | 1.705 |
| **R² Coefficient** | 0.054 |

### 2. Official Ranking Power
| Metric | Value |
| :--- | :--- |
| **Mean Spearman Correlation** | 0.174 |
| **Mean Kendall Tau Correlation** | 0.123 |
| **Mean Predictive Index (PI)** | 0.164 |
| **Top-1 Success Rate** | 35.09% |
| **Top-2 Success Rate** | 52.63% |
| **Top-3 Success Rate** | 73.68% |

### 3. Uncertainty & Calibration
| Metric | Value |
| :--- | :--- |
| **Expected Calibration Error (ECE)** | 0.3371 |
| **Negative Log-Likelihood (NLL)** | 8.0741 |
| **PICP @ 95% Confidence** | 36.84% |
| **MPIW @ 95% Confidence** | 2.149 |

## Figures & Plot Archives
All 10 validation plots are exported in 300 DPI high-resolution PNG, vector PDF, and editable SVG formats under `reports/CASF_Evaluation/figures/`.
