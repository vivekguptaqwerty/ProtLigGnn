# 08. Error Analysis Report
**Experiment ID**: P5.2-UNCERTAINTY  

## Selective Prediction Rejection Analysis
We evaluate predictive performance gains as higher-uncertainty complexes are rejected.

### Selective Rejection Matrix
| Rejection Fraction | Coverage | Remaining Sample Count | RMSE | Relative Improvement (%) |
| :--- | :---: | :---: | :---: | :---: |
| **0% Rejection** | 1.00 | 50 | 1.5022 | 0.0% |
| **5% Rejection** | 0.95 | 47 | 1.4112 | 6.0% |
| **10% Rejection** | 0.90 | 45 | 1.3214 | 12.0% |
| **20% Rejection** | 0.80 | 40 | 1.1524 | 23.2% |

*Interpretation*: Filtering predictions by uncertainty leads to systematic improvements in prediction quality on the remaining sample coverage.
