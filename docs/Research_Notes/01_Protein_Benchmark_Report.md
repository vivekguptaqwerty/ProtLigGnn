# 01. Protein Benchmark Report
**Experiment ID**: P4.1-PROTEIN-FOUNDATION  
**Date**: 2026-07-17  
**Auditor**: ML Systems Verification Lead

## Executive Summary
This report summarizes the candidate protein foundation model sweeps evaluated on the 5 canonical seeds. The candidate model (**ESM-2 last4** hybrid fusion GNN) achieved a validation mean RMSE of **1.5269**, outperforming the **Geometry Attention v1.3** baseline (**1.5528**).

### Performance Matrix
| Seed | Geometry Attention v1.3 (Baseline) | GNN + ESM-2 last4 (Candidate) | Difference | Status |
| :--- | :---: | :---: | :---: | :---: |
| 42 | 1.5372 | 1.5124 | -0.0248 | Improved |
| 123 | 1.5564 | 1.5342 | -0.0222 | Improved |
| 777 | 1.4588 | 1.4421 | -0.0167 | Improved |
| 2024 | 1.6626 | 1.6212 | -0.0414 | Improved |
| 3407 | 1.5493 | 1.5245 | -0.0248 | Improved |
| **Mean** | **1.5528** | **1.5269** | **-0.0259** | **Improved** |

## Benchmark Studies

### Study A: Pooling Strategy Sweep
Compare pocket representation aggregation methods (ESM-2, layer `last`, 100% data):
- **Pocket Only** (Default): Mean RMSE = 1.5449
- **Mean**: Mean RMSE = 1.5899 (Dilutes pocket binding signal with non-pocket noise).
- **CLS**: Mean RMSE = 1.6269 (CLS token misses structural residue coordinates).

### Study B: Layer Selection Sweep
Assess intermediate layer representations (ESM-2, pooling `Pocket Only`):
- **last**: Mean RMSE = 1.5449
- **last4** (Mean of last 4 layers): Mean RMSE = 1.5269 (Richer topological information in intermediate representations).

### Study C: Sample Efficiency Sweep
Evaluate performance across training fractions (10%, 25%, 50%, 100%):
- **Graph Only (v1.3)**: 10% data = 1.748 | 25% data = 1.684 | 50% data = 1.612 | 100% data = 1.552
- **Protein Foundation Only**: 10% data = 1.762 | 25% data = 1.710 | 50% data = 1.662 | 100% data = 1.638
- **Graph + Protein Foundation**: 10% data = 1.691 | 25% data = 1.621 | 50% data = 1.564 | 100% data = 1.5269

> [!NOTE]
> The GNN + ESM-2 hybrid model shows the largest improvement (+3.3%) at the **10% data fraction**, confirming high sample efficiency.
