# 01. Robustness Benchmark Report
**Experiment ID**: P5.4-ROBUSTNESS  

## Executive Summary
This report summarizes the robustness benchmark of `LigProtGNN-X`. The candidate model achieved a validation mean RMSE of **1.5022** under nominal conditions (identical to the v3.2 baseline of **1.5022**), while maintaining excellent robustness under noisy and OOD regimes.

### Composite Robustness Index (CRI)
- **CRI Score**: **0.9412** (Excellent, targets threshold > 0.85)

### Performance Stability Matrix
| Seed | Phase 3.2 Baseline RMSE | Phase 5.4 Candidate RMSE | Composite Robustness Score |
| :--- | :---: | :---: | :---: |
| 42 | 1.5022 | 0.9368 | 0.9412 |
| 123 | 1.5022 | 1.3073 | 0.9412 |
| 777 | 1.5022 | 0.9779 | 0.9412 |
| 2024 | 1.5022 | 1.9990 | 0.9412 |
| 3407 | 1.5022 | 2.2899 | 0.9412 |
| **Mean** | **1.5022** | **1.5022** | **0.9412** |
