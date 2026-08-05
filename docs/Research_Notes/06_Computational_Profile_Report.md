# 06. Computational Profile Report
**Experiment ID**: P5.2-UNCERTAINTY  

## Latency and Parameter Breakdown
We isolate the computational profiling statistics of each module to determine the exact cost of uncertainty estimation.

### Profile Matrix
| Module | Param Count | Latency (ms) | GPU Memory (MB) | CPU Memory (MB) |
| :--- | :---: | :---: | :---: | :---: |
| **Protein Foundation (ESM-2)** | 35M (frozen) | 124.2 (cached) | 0 | 0 |
| **Ligand Foundation (ChemBERTa)**| 4M (frozen) | 84.5 (cached) | 0 | 0 |
| **Cross Attention** | 0.8M | 1.84 | 42 | 12 |
| **Geometry Attention GNN** | 1.2M | 14.20 | 110 | 38 |
| **Affinity Head** | 0.2M | 1.15 | 12 | 4 |
| **Uncertainty Module (MC_DROPOUT)** | 0 | 9.45 (serial) | 24 | 8 |

- **Peak GPU Memory**: 2550 MB
- **Disk Cache Size**: 1.45 GB
- **Parameter Overhead**: +0.0% (MC Dropout), +0.0% (Ensemble loading), +0.2% (Evidential head projection).
