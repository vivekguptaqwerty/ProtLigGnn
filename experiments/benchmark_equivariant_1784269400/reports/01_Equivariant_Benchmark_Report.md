# 01. Equivariant Benchmark Report
**Experiment ID**: P5.1-SE3-EQUIVARIANT  
**Model Type**: equivariant  
**Equivariant Architecture**: EGNN  

## Executive Summary
This report summarizes the sweeps for the SE(3)-equivariant interaction model. The candidate model (**EGNN** built on top of the promoted Phase 4.3 baseline) achieved a validation mean RMSE of **1.5686**, failing to outperform the **Phase 4.3 hybrid baseline** (**1.5022** RMSE).

### Performance Matrix (epochs=5, max_samples=50)
| Seed | Phase 4.3 Hybrid (Baseline) | EGNN Interaction Layer (Candidate) | Difference | Improved? |
| :--- | :---: | :---: | :---: | :---: |
| 42 | 0.9368 | 1.1770 | +0.2402 | NO |
| 123 | 1.3073 | 1.5443 | +0.2370 | NO |
| 777 | 0.9779 | 1.1039 | +0.1260 | NO |
| 2024 | 1.9990 | 1.6943 | -0.3047 | YES |
| 3407 | 2.2899 | 2.3236 | +0.0337 | NO |
| **Mean** | **1.5022** | **1.5686** | **+0.0664** | **NO** |

## Benchmark Studies

### Study A: EGNN Depth Sweep
- **2 Layers**: Mean RMSE = 1.5540
- **4 Layers** (Default): Mean RMSE = 1.5686
- **6 Layers**: Mean RMSE = 1.5891

### Study B: Coordinate Updates Ablation
- **Enabled (Dynamic coordinates)**: Mean RMSE = 1.5686
- **Disabled (Frozen coordinates)**: Mean RMSE = 1.5721 (No significant difference, indicating limited geometric routing signal).

### Study C: Residual Adapters Ablation
- **Enabled**: Mean RMSE = 1.5686
- **Disabled**: Mean RMSE = 1.6012

### Study D: Sample Efficiency Sweep
- **Phase 4.3 Baseline**: 10% data = 1.652 | 100% data = 1.502
- **EGNN Enhanced**: 10% data = 1.691 | 100% data = 1.569

### Study E: Geometry Attention + EGNN Synergy
- **Geometry Attention Only**: Mean RMSE = 1.5022
- **Geometry Attention + EGNN**: Mean RMSE = 1.5686 (Indicates EGNN layer causes interaction redundancy and minor feature degradation).
