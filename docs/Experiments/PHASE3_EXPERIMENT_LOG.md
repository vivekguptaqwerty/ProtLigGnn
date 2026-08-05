# Phase 3 Controlled Experiment Log
**File Status**: Active  

This log records detailed scientific entries for all completed Phase 3 experiments.

---

## Experiment Entry: P3.1A-GEOMETRY-LIGAND

- **Experiment ID**: `P3.1A-GEOMETRY-LIGAND`
- **Branch**: `main`
- **Date**: 2026-07-12
- **Independent Variable**: Ligand geometric edge representations (Euclidean distances mapped via Gaussian RBF expansion).
- **Controlled Variables**:
  - Protein representation (GCNConv, coordinate-free pocket graph).
  - Cross-attention projection (4 heads, coordinate-free, identical to Baseline v1.1).
  - Regressor block (MLP, identical to Baseline v1.1).
  - Optimizer: AdamW, Scheduler: CosineAnnealingLR, Epochs: 40, Batch Size: 8, Data splits and seeds.
- **Benchmark Status**: Completed.
- **Statistical Outcome**: 
  - Mean RMSE reduced from $1.6048$ to $1.5442$ (a $3.77\%$ improvement).
  - Paired t-test yields $p = 0.1484$, which is larger than the $\alpha = 0.05$ significance threshold. The null hypothesis $H_0$ is retained.
- **Promotion Decision**: **REJECTED (Retained v1.1)**. The candidate model is not promoted because the improvement is not statistically significant. Baseline v1.1 remains the active baseline.
- **Lessons Learned**:
  1. **Spatial Representation Isolation**: Upgrading the ligand encoder with geometry features without changing the interaction layer (cross-attention) does not propagate spatial constraints across the binding pocket interface. The cross-attention coefficients remain weakly correlated with physical atom-residue distances ($r \le 0.16$).
  2. **Statistical Power Constraints**: A 5-seed benchmark group has low statistical power. Large effect sizes (Cohen's d of $\approx -0.80$ for RMSE and $\approx -1.19$ for MAE) fail to reach statistical significance.
- **Next Experiment (Phase 3.1B)**: Introduce **distance-biased cross-attention** where cross-attention weights are explicitly biased by physical ligand-residue Euclidean distances to force spatial interaction learning.
