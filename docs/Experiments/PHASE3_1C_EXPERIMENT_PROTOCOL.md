# Phase 3.1C Experiment Protocol
**Experiment ID**: `P3.1C-EDGE-AWARE-INTERACTION`  
**Status**: Proposed  

This protocol defines the variables, datasets, training configurations, and evaluation steps to ensure rigorous, reproducible comparisons.

---

## 1. Experimental Variables

### 1.1 Independent Variable
The architecture of the protein-ligand interaction block:
- **Reference**: Geometry Attention v1.3 (`--model_type geometry_attention`) — scalar distance logit bias.
- **Candidate**: Geometry Interaction v1.4 (`--model_type geometry_interaction`) — multi-branch Interaction Representation Module with Concatenation Fusion and Linear logit projection.

### 1.2 Dependent Variables
- **Regression Accuracy**: test set RMSE, MAE, and coefficient of determination ($R^2$).
- **Ranking Quality**: Pearson Correlation Coefficient (PCC) and Spearman's rank correlation ($\rho$).
- **Spatial Alignment & Localization**: Pearson correlation of attention weights vs $1/D$, normalized attention entropy, Top-K localization precision.
- **Computational Overhead**: parameter count, forward latency, peak GPU VRAM, checkpoint size.

### 1.3 Controlled Variables (Frozen)
The following hyperparameters and split settings are permanently locked:
- **Dataset**: PDBbind 2020 local subset (1,000 complexes, SHA-256 `6fb4385c2...`).
- **Data Splits**: 5-seed random splits (`[42, 123, 777, 2024, 3407]`).
- **Optimizer**: AdamW ($LR = 1e-3$, weight decay $1e-4$).
- **Learning Rate Scheduler**: CosineAnnealingLR (minimum $LR = 1e-5$, 30 epochs).
- **Epochs**: 30 epochs.

---

## 2. Evaluation Steps

1. **Pre-Benchmark Validation**: Complete all unit tests and run a 2-epoch CPU smoke check.
2. **Benchmark Execution**: Train candidate models across the 5 canonical seeds and save predictions.
3. **Statistical significance test**: Perform paired t-test over the 5 seed RMSE results against Baseline v1.3.
4. **Promotion evaluation**: Trigger promotion to Baseline v1.4 if and only if RMSE improves with $p < 0.05$.
