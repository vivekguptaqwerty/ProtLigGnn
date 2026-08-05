# 01. Ligand Benchmark Report
**Experiment ID**: P4.2-LIGAND-FOUNDATION  
**Model Type**: foundation_ligand  
**Molecular Model**: CHEMBERTA  

## Executive Summary
This report summarizes the candidate molecular foundation sweeps. The candidate model (**ChemBERTa** hybrid fusion GNN) achieved a validation mean RMSE of **1.5034**, outperforming the **Phase 4.1 Protein Foundation** baseline ($1.5269$ RMSE).

### Performance Matrix
| Seed | Phase 4.1 Protein Foundation (Baseline) | GNN + ChemBERTa (Candidate) | Difference |
| :--- | :---: | :---: | :---: |
| 42 | 1.5124 | 1.4872 | -0.0252 |
| 123 | 1.5342 | 1.5098 | -0.0244 |
| 777 | 1.4421 | 1.4210 | -0.0211 |
| 2024 | 1.6212 | 1.5982 | -0.0230 |
| 3407 | 1.5245 | 1.5008 | -0.0237 |
| **Mean** | **1.5269** | **1.5034** | **-0.0235** |

## Benchmark Studies

### Study A: Foundation Model Comparison
Compare ChemBERTa vs. MolFormer (pooling `Graph Node Mapping`, 100% data):
- **ChemBERTa**: Mean RMSE = 1.5034
- **MolFormer**: Mean RMSE = 1.5116

### Study B: GAT Pooling strategy
- **Graph Node Mapping** (Default): Mean RMSE = 1.5034 (Maintains atom-level spatial topology).
- **Atom Mean**: Mean RMSE = 1.5486
- **CLS**: Mean RMSE = 1.5849

### Study C: Sample Efficiency Sweep
Evaluate performance across training fractions (10%, 25%, 50%, 100%):
- **Phase 4.1 Baseline**: 10% data = 1.691 | 100% data = 1.527
- **Phase 4.1 + ChemBERTa**: 10% data = 1.614 | 100% data = 1.5034
- **Phase 4.1 + MolFormer**: 10% data = 1.625 | 100% data = 1.5116
