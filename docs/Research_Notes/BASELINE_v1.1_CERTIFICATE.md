# Constitutional Certificate of Validation & Repository Freeze
## ProtLigGNN Optimized Baseline v1.1

**Certificate ID**: `PLGNN-CERT-20260712-V1.1`  
**Certification Date**: 2026-07-12  
**Maturity Classification**: Frozen Reference Platform  

---

## 1. Repository Identity & Metadata

- **Codebase Path**: `D:\ProtLigGnn`
- **Certified Git Commit**: `8a90ecaca59d5859191d24badc304efd778eb1f4`
- **Official Model Target**: Optimized Baseline v1.1
- **Current Milestone**: Phase 2 Scientific Hardening & certification
- **License/Maintainer**: Principal deep learning Engineer & Long-Term Maintainer

---

## 2. Hardware & Environment Fingerprints

- **Execution Environment**: ProtLigGNN Environment v1.0 (Python 3.11.9)
- **Primary PyTorch Version**: `2.3.1+cu121`
- **PyTorch Geometric (PyG) Version**: `2.8.0`
- **RDKit Version**: `2024.03.1`
- **CUDA Device Available**: NVIDIA GPU CUDA 12.1 Enabled

---

## 3. Dataset & Graph Cache Fingerprints

- **Target Dataset**: PDBbind v2020 Local Subset
- **Pre-processed Sample Count**: 1,000 complexes
- **Graph Cache File**: `data/pdbbind2020/processed_dataset.pt`
- **Cache File Size**: 32,854,946 bytes
- **Cache SHA-256 Checksum**: `6fb4385c2136b197ef95501c4eea8775ad485702bfa5391b8cdbda909a108a3f`
- **Dataset Hash (Manifest)**: `41a58b48396fb781c073b1a908fc03a353f144f6a434c8c1b36910bc059ec5b1`
- **Feature Schema Version**: 1.0 (Ligand atom features: 9 nodes/78 dim, Protein residue features: 30 dim)

---

## 4. Certified Model Architecture

The official architecture for **Optimized Baseline v1.1** is defined as:
- **Ligand Encoder**: GATConv (2 layers, hidden_dim=128, dropout=0.3181)
- **Protein Encoder**: GCNConv (2 layers, hidden_dim=128, dropout=0.3181)
- **Interaction Layer**: BidirectionalCrossAttention (hidden_dim=128, 4 attention heads)
- **Pooling Layer**: Global Mean Pooling
- **Regression Head**: MLP (Layers: [256, 128, 64, 1])
- **Optimizer**: AdamW (lr=0.00019, weight_decay=0.000117, grad_clip=1.0)
- **Scheduler**: CosineAnnealingLR (epochs=40, patience=10)

---

## 5. Certified Benchmark Metrics

The baseline model has been evaluated across the **5 official benchmark seeds** using 10% validation and 10% test splits. The aggregated test set metrics are:

| Seed | RMSE | MAE | PCC | Spearman | $R^2$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | 1.5863 | 1.2162 | 0.4342 | 0.4662 | 0.1728 |
| **123** | 1.6286 | 1.2603 | 0.5495 | 0.4979 | 0.2348 |
| **777** | 1.5139 | 1.2749 | 0.5352 | 0.5297 | 0.2616 |
| **2024** | 1.6659 | 1.3065 | 0.5501 | 0.5348 | 0.2750 |
| **3407** | 1.6291 | 1.2845 | 0.4628 | 0.4269 | 0.0590 |
| **Mean** | **1.6048** | **1.2685** | **0.5064** | **0.4911** | **0.2006** |
| **Std** | **0.0520** | **0.0301** | **0.0484** | **0.0405** | **0.0791** |

---

## 6. Known Scientific Limitations

1. **Regression to the Mean**: The model exhibits a slope of $-0.66$ ($r=-0.75$) between residuals and true affinities. It systematically over-predicts weak binders (mean residual $+1.64$ for $<4.5$) and under-predicts strong binders (mean residual $-1.72$ for $>7.5$).
2. **Implicit Geometry**: The model encodes structural details as a coordinate-free graph with implicit spatial distances, leading to poor explicit distance-attention correlations.
3. **Local Pocket Restriction**: Pocket extraction is limited to residues within a 12Å boundary of the ligand.

---

## 7. Rules for Phase 3 Research

1. **Frozen Platform**: No modifications to `protliggnn_train.py`, `benchmarking.py`, or `data/` are allowed under Phase 2 namespace.
2. **Equitable Evaluation**: Every new GNN architecture or representation learning module must be evaluated against this exact configuration using the same seeds, cache, splits, and metrics.
3. **No Baseline Promotion without Paired T-Test**: A new candidate is promoted only if its mean RMSE is statistically lower than 1.6048 with a paired t-test p-value $< 0.05$ across the 5 seeds.

---

## 8. Declaration of Freeze

This repository is hereby certified.

Optimized Baseline v1.1 is the official immutable reference implementation.

Every future architectural contribution must be evaluated against this certified baseline.
