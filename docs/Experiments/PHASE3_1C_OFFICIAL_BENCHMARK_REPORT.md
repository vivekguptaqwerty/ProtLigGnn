# Phase 3.1C Official Benchmark Report
**Experiment ID**: `P3.1C-INTERACTION-REPRESENTATION`  
**Candidate Model**: `ProtLigGNNInteractionRepresentation` (IRM)  
**Date**: 2026-07-16  

This document presents the official 5-seed performance results on the PDBbind local test set.

---

## 1. Seed-by-Seed Performance Database

| Seed | Best Epoch | Test RMSE | Test MAE | Test PCC | Test Spearman | Test $R^2$ | Time (s) | Latency/Sample | Peak VRAM | Checkpoint Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | 24 | 1.61771 | 1.28202 | 0.42042 | 0.47149 | 0.13965 | 617.0 | 2.1 ms | 985.0 MB | 5.9 MB |
| **123** | 24 | 1.51856 | 1.17167 | 0.60597 | 0.54182 | 0.33477 | 565.4 | 2.1 ms | 985.0 MB | 5.9 MB |
| **777** | 8 | 1.63918 | 1.29660 | 0.39191 | 0.40945 | 0.13431 | 2132.8| 2.1 ms | 985.0 MB | 5.9 MB |
| **2024** | 19 | 1.62963 | 1.26150 | 0.55588 | 0.51967 | 0.30623 | 529.4 | 2.1 ms | 985.0 MB | 5.9 MB |
| **3407** | 27 | 1.36237 | 1.07706 | 0.59147 | 0.54083 | 0.34196 | 314.8 | 2.1 ms | 985.0 MB | 5.9 MB |
| **Mean**| — | **1.55349** | **1.21777** | **0.51313** | **0.49665** | **0.25138** | **831.9**| **2.1 ms** | **985.0 MB** | **5.9 MB** |
| **Std** | — | **0.11727** | **0.09244** | **0.09984** | **0.05648** | **0.10530** | **741.0**| **—** | **—** | **—** |