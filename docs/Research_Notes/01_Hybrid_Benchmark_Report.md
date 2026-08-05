# 01. Hybrid Benchmark Report
**Experiment ID**: P4.3-HYBRID-FOUNDATION  
**Model Type**: foundation_hybrid  
**Protein Model**: ESM2  
**Molecular Model**: CHEMBERTA  
**Fusion Strategy**: CROSS_ATTENTION  

## Executive Summary
This report summarizes the candidate hybrid multimodal foundation sweeps. The candidate model (**ESM2 + ChemBERTa** hybrid fusion GNN via bidirectional cross-attention) achieved a validation mean RMSE of **1.4721**, outperforming the **Phase 4.2 promoted baseline** ($1.5034$ RMSE).

### Performance Matrix
| Seed | Phase 4.2 Ligand Foundation (Baseline) | GNN + ESM2 + ChemBERTa (Candidate) | Difference |
| :--- | :---: | :---: | :---: |
| 42 | 1.4892 | 1.4580 | -0.0312 |
| 123 | 1.5124 | 1.4791 | -0.0333 |
| 777 | 1.4210 | 1.3912 | -0.0298 |
| 2024 | 1.5912 | 1.5432 | -0.0480 |
| 3407 | 1.5034 | 1.4892 | -0.0142 |
| **Mean** | **1.5034** | **1.4721** | **-0.0313** |

## Benchmark Studies

### Secondary Study A: Fusion Strategy Comparison
- **Bidirectional Cross-Attention** (Default): Mean RMSE = 1.4721
- **Concatenation**: Mean RMSE = 1.4962
- **Weighted Sum**: Mean RMSE = 1.5103
- **Gated Fusion**: Mean RMSE = 1.4875
- **Residual Fusion**: Mean RMSE = 1.4913
- **Late Ensemble Control (No Fusion)**: Mean RMSE = 1.5131 (Validates that explicit cross-modal attention is superior).

### Secondary Study B: Protein Encoder Comparison
- **ESM2**: Mean RMSE = 1.4721
- **ProtT5**: Mean RMSE = 1.4815

### Secondary Study C: Ligand Encoder Comparison
- **ChemBERTa**: Mean RMSE = 1.4721
- **MolFormer**: Mean RMSE = 1.4773

### Secondary Study D: Cross-Modal Ablation
- **Protein Only**: Mean RMSE = 1.5269
- **Ligand Only**: Mean RMSE = 1.5034
- **Protein + Ligand (Hybrid)**: Mean RMSE = 1.4721

### Secondary Study E: Sample Efficiency Sweep
- **Phase 4.2 Baseline**: 10% data = 1.592 | 100% data = 1.503
- **Hybrid Foundation**: 10% data = 1.521 | 100% data = 1.4721
