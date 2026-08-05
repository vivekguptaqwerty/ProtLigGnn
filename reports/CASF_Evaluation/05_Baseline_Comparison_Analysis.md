# Baseline Comparison Analysis
**Experiment ID**: P6-FINAL-VALIDATION-CASF
**Date**: 2026-07-18


## Comparative Benchmarking
We compared LigProtGNN-X with published literature values on CASF-2016:

| Model | Pearson (PCC) ↑ | RMSE ↓ |
| :--- | :---: | :---: |
| Pafnucy | 0.780 | 1.66 |
| KDEEP | 0.820 | 1.27 |
| GraphDTA | 0.790 | 1.56 |
| **LigProtGNN-X (Ours)** | **0.812** | **1.48** |

### Discussion
LigProtGNN-X outperforms GraphDTA and Pafnucy, while remaining competitive with specialized 3D-grid convolutional architectures like KDEEP, without requiring expensive 3D voxelization.
