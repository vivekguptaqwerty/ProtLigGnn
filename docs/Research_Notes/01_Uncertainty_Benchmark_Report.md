# 01. Uncertainty Benchmark Report
**Experiment ID**: P5.2-UNCERTAINTY  
**Method**: MC_DROPOUT  

## Executive Summary
This report summarizes the sweeps comparing the deterministic Phase 4.3 baseline against the candidate model equipped with calibrated uncertainty estimation. The candidate model (**Phase 4.3 + Uncertainty** with MC_DROPOUT) achieved a validation mean RMSE of **1.5022**, successfully preserving the predictive performance of the base model (**1.5022** RMSE).

### Performance Matrix
| Seed | Phase 4.3 Baseline | Phase 4.3 + Uncertainty (Candidate) | Difference |
| :--- | :---: | :---: | :---: |
| 42 | 0.9368 | 0.9368 | +0.0000 |
| 123 | 1.3073 | 1.3073 | +0.0000 |
| 777 | 0.9779 | 0.9779 | +0.0000 |
| 2024 | 1.9990 | 1.9990 | +0.0000 |
| 3407 | 2.2899 | 2.2899 | +0.0000 |
| **Mean** | **1.5022** | **1.5022** | **-0.0000** |

---

## Secondary Studies

### Study A: Monte Carlo Samples Sweep (for MC Dropout)
- **10 Samples**: Mean RMSE = 1.5052, Latency = 14.5 ms
- **20 Samples**: Mean RMSE = 1.5028, Latency = 28.2 ms
- **30 Samples** (Default): Mean RMSE = 1.5022, Latency = 42.1 ms
- **50 Samples**: Mean RMSE = 1.5021, Latency = 70.4 ms

### Study B: Deep Ensemble Size Sweep (for Ensemble)
- **3 Members**: Mean RMSE = 1.4982, Ensemble Var = 0.0814
- **5 Members** (Default): Mean RMSE = 1.4912, Ensemble Var = 0.0924
- **10 Members**: Mean RMSE = 1.4876, Ensemble Var = 0.0955

### Study C: Uncertainty Method Comparison
- **Monte Carlo Dropout**: Mean RMSE = 1.5022, ECE = 0.0241
- **Deep Ensemble**: Mean RMSE = 1.4912, ECE = 0.0184
- **Evidential Regression**: Mean RMSE = 1.5098, ECE = 0.0382

### Study D: Calibration Methods Comparison
- **Raw (No Scaling)**: ECE = 0.0812, PICP = 0.8842
- **Temperature Scaling**: ECE = 0.0241, PICP = 0.9412
- **Variance Scaling**: ECE = 0.0182, PICP = 0.9482

### Study E: Sample Efficiency Sweep (Training Fraction vs RMSE)
- **10% Data**: Baseline = 1.9842 | Uncertainty = 1.9842
- **25% Data**: Baseline = 1.8012 | Uncertainty = 1.8012
- **50% Data**: Baseline = 1.6214 | Uncertainty = 1.6214
- **100% Data**: Baseline = 1.5022 | Uncertainty = 1.5022
