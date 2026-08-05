# Phase 3.1C Representation Analysis Report
**Experiment ID**: `P3.1C-INTERACTION-REPRESENTATION`  
**Date**: 2026-07-16  

This report analyzes the latent embedding properties of the IRM.

---

## 1. Branch Contributions (Projector Weights)
Analyzing the absolute weights of the `ConcatFusion` projection layer:
- **GNN Node Representations**: **48.30%** (Dominant branch)
- **Chemistry Encoder**: **37.27%**
- **Geometry RBF Encoder**: **14.43%**

---

## 2. Latent Space Statistics
- **Mean Embedding Norm**: \(4.5372 \pm 0.0003\)
- **Active Dimensions**: \(32 / 32\) (Under random inputs, GNN node values yield complete latent space utilization)