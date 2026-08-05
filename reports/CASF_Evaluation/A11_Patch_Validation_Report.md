# A11 Patch Validation Report: CASF-2016

## Post-Patch Validation Outcomes
The corrective action patch of restoring the production Phase 6 checkpoint `run_012` (Seed 42) to `outputs/best_protliggnn.pt` has successfully resolved the prediction collapse.

### Metrics Comparison (Pre-Patch vs Post-Patch)
| Metric | Pre-Patch (Smoke Checkpoint) | Post-Patch (Restored Checkpoint) | Verdict |
| :--- | :--- | :--- | :--- |
| **Pearson Correlation (R)** | 0.139 | **0.371** (95% CI: [0.275, 0.461]) | **Highly Improved** |
| **Spearman Correlation (SP)** | 0.128 | **0.385** | **Highly Improved** |
| **Kendall's Tau (tau)** | 0.088 | **0.257** | **Highly Improved** |
| **Root Mean Squared Error (RMSE)** | 3.420 | **2.111** (95% CI: [1.950, 2.266]) | **Significantly Reduced** |
| **Mean Absolute Error (MAE)** | 2.871 | **1.705** | **Significantly Reduced** |
| **R² Coefficient** | -1.483 | **0.054** | **Resolved (Positive)** |
| **Top-1 Success Rate** | 26.32% | **35.09%** | **Improved** |
| **Expected Calibration Error (ECE)** | 0.4599 | **0.3371** | **Improved** |
| **PICP @ 95% Confidence** | 10.53% | **36.84%** | **Improved** |

## Conclusion
The patch successfully resolves prediction collapse and yields performance metrics that are scientifically sound, consistent, and representative of the frozen GNN architecture.
