# Official Benchmark Report: Phase 3.1C
**Experiment ID**: `P3.1C-EDGE-AWARE-INTERACTION`  
**Candidate Model**: Geometry Interaction v1.4 (`--model_type geometry_interaction`)  
**Status**: **COMPLETED**  
**Date**: 2026-07-15  

This document reports the performance of the candidate Geometry Interaction model v1.4 across the 5 canonical seeds.

---

## 1. Benchmark Execution Environment
- **Hardware**: Nvidia RTX 3050 (6GB VRAM), AMD Ryzen 7
- **Software**: Python 3.11.9, PyTorch 2.3.1+cu121, PyTorch Geometric 2.5.3
- **Dataset Cache Hash**: `6fb4385c2136b197ef95501c4eea8775ad485702bfa5391b8cdbda909a108a3f`
- **Seeds**: `42`, `123`, `777`, `2024`, `3407`

---

## 2. Seed-by-Seed Performance

The five benchmark runs were executed independently on GPU for 40 epochs:

| Seed | Best Epoch | Test RMSE | Test MAE | Test PCC | Test Spearman | Test $R^2$ | Checkpoint Path |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **42** | 24 | 1.61771 | 1.28202 | 0.42042 | 0.47149 | 0.13965 | `experiments/run_061/checkpoint.pt` |
| **123** | 24 | 1.51856 | 1.17167 | 0.60597 | 0.54182 | 0.33477 | `experiments/run_062/checkpoint.pt` |
| **777** | 8 | 1.63918 | 1.29660 | 0.39191 | 0.40945 | 0.13431 | `experiments/run_063/checkpoint.pt` |
| **2024** | 19 | 1.62963 | 1.26150 | 0.55588 | 0.51967 | 0.30623 | `experiments/run_064/checkpoint.pt` |
| **3407** | 27 | 1.36237 | 1.07706 | 0.59147 | 0.54083 | 0.34196 | `experiments/run_065/checkpoint.pt` |
| **Mean**| — | **1.55349** | **1.21777** | **0.51313** | **0.49665** | **0.25138** | — |
| **Std** | — | **0.11727** | **0.09244** | **0.09984** | **0.05648** | **0.10530** | — |

---

## 3. Comparative Summary

| Phase / Architecture | Mean RMSE | Mean MAE | Mean PCC | Mean Spearman | Mean $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline v1.1** | $1.60476 \pm 0.05808$ | $1.26848 \pm 0.03370$ | $0.50636 \pm 0.05411$ | $0.49110 \pm 0.04524$ | $0.20064 \pm 0.08839$ |
| **Geometry v1.2** (3.1A) | $1.54426 \pm 0.08180$ | $1.21392 \pm 0.06282$ | $0.54704 \pm 0.01972$ | $0.54370 \pm 0.02935$ | $0.26350 \pm 0.03244$ |
| **Attention v1.3** (3.1B) | $1.55288 \pm 0.07271$ | $1.21532 \pm 0.03289$ | $0.53765 \pm 0.03406$ | $0.53033 \pm 0.03638$ | $0.25310 \pm 0.06786$ |
| **IRM v1.4** (3.1C) | **$1.55349 \pm 0.11727$** | **$1.21777 \pm 0.09244$** | **$0.51313 \pm 0.09984$** | **$0.49665 \pm 0.05648$** | **$0.25138 \pm 0.10530$** |
