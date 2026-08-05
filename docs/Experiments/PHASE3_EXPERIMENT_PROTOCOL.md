# Phase 3.1A Experiment Protocol
**Experiment ID**: P3.1A-GEOMETRY-LIGAND  
**Target Model**: ProtLigGNNGeometry  
**Baseline Model**: ProtLigGNN Optimized Baseline v1.1  
**Status**: Signed and Certified  

This document defines the experimental protocol for testing the effect of explicit ligand geometry on ProtLigGNN. It follows publication-quality guidelines for scientific ML research.

---

## 1. Experimental Variables

### 1.1 Independent Variable
- **Explicit Invariant Ligand Geometry**: Covalent bond Euclidean distances mapped using 32 Gaussian RBF kernels, projected to 32 dimensions, and consumed as edge attributes by GATConv layers in the ligand encoder.

### 1.2 Dependent Variables
1. **RMSE (Primary)**: Root Mean Squared Error on the test set.
2. **MAE (Secondary)**: Mean Absolute Error on the test set.
3. **PCC (Secondary)**: Pearson Correlation Coefficient.
4. **Spearman (Secondary)**: Spearman Rank Correlation Coefficient.
5. **$R^2$ (Secondary)**: Coefficient of Determination.
6. **Attention-Distance Correlation**: Correlation between cross-attention weights and physical distance.

### 1.3 Controlled Variables (Frozen)
- **Dataset**: PDBbind v2020 subset ($N=1,000$).
- **Graph Cache**: `data/pdbbind2020/processed_dataset.pt` (SHA-256: `6fb4385c2...`).
- **Splits**: 80/10/10 random splits based on frozen split IDs.
- **Seeds**: `[42, 123, 777, 2024, 3407]`.
- **Training Protocol**: 40 epochs max, Cosine Annealing learning rate schedule, AdamW optimizer (lr=0.00019, weight_decay=0.000117), batch size=8.
- **Hardware/Environment**: PyTorch 2.3.1+cu121, PyG 2.8.0, NVIDIA GPU execution.

---

## 2. Statistical Analysis & Testing
- **Paired T-Test**: To reject the null hypothesis $H_0$, we will perform a two-tailed paired t-test comparing the test RMSE values of `ProtLigGNNGeometry` against `ProtLigGNN` across the 5 canonical seeds.
- **Significance Threshold**: $\alpha = 0.05$. We reject $H_0$ only if the t-test yields $p < 0.05$.

---

## 3. Threats to Validity

### 3.1 Internal Validity
- *Threat*: Initialization variances or gradient instability could skew results.
- *Mitigation*: We evaluate across 5 canonical seeds with fixed PyTorch and NumPy seed initialization. Gradient clipping is set to 1.0 to prevent gradient explosions.

### 3.2 External Validity
- *Threat*: The local subset of 1,000 complexes may not generalize to the entire PDBbind dataset or other screening targets.
- *Mitigation*: This is a standard constraint of the frozen benchmark. Results are interpreted as a proof-of-concept for the representation capacity.

### 3.3 Construct Validity
- *Threat*: Differences in parameter sizes could account for performance improvements rather than explicit geometry.
- *Mitigation*: The addition of geometry projections introduces only 1,120 parameters (0.3% increase). We verify that model capacity remains equivalent.

### 3.4 Statistical Validity
- *Threat*: Running only 5 seeds may result in low statistical power for the t-test.
- *Mitigation*: The 5 seeds are canonically defined by the repository freeze. To increase confidence, we report the standard deviation and confidence intervals alongside p-values.

---

## 4. Promotion Criteria
For the candidate model `ProtLigGNNGeometry` to be promoted to **Baseline v2.0**, it must satisfy:
1. **Performance superiority**: Mean RMSE across the 5 seeds must be lower than Baseline v1.1 ($1.6048$) with a paired t-test $p < 0.05$.
2. **Reproduction checklist**: Must pass all verification tests (`pytest tests/`) with zero failures.
3. **No Baseline Regressions**: Running `--model_type baseline` must produce identical predictions and parameter counts to Baseline v1.1.
