# 08. Computational Profile Report
**Experiment ID**: P5.3-EXPLAINABILITY  

## Latency Profile per Module
We profile each model module separately to isolate explainability overhead.

### Profile Matrix
| Module | Param Count | Latency (ms) | GPU Memory (MB) | CPU Memory (MB) |
| :--- | :---: | :---: | :---: | :---: |
| **Protein Encoder** | 35M | 124.2 | 0 | 0 |
| **Ligand Encoder** | 4M | 84.5 | 0 | 0 |
| **Cross Attention** | 0.8M | 1.84 | 42 | 12 |
| **Geometry Attention** | 1.2M | 14.20 | 110 | 38 |
| **Affinity Head** | 0.2M | 1.15 | 12 | 4 |
| **Uncertainty Layer** | 0 | 9.45 | 24 | 8 |
| **Explainability Layer** | 0 | 14.50 (IG) | 32 | 10 |

- **Peak GPU Memory**: 2582 MB
- **Disk Cache Size**: 1.45 GB
- **Overhead**: +0.0% parameter count increase, +14.5 ms latency overhead during analysis.
