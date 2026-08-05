# Phase 3.1A Official Experiment Certification
**Experiment ID**: P3.1A-GEOMETRY-LIGAND  
**Certified Date**: 2026-07-12  
**Certification Status**: **CERTIFIED EQUIVALENT (Not Promoted)**  

---

## 1. Core Experiment Specification
1. **Experiment ID**: `P3.1A-GEOMETRY-LIGAND`
2. **Scientific Question**: Does explicit ligand geometric edge representations (Euclidean distances mapped via Gaussian RBF) improve graph representation quality in the ligand encoder and result in better protein-ligand affinity predictions?
3. **Independent Variable**: Ligand geometric edge representations.
4. **Controlled Variables**:
   - Protein pocket encoder (GCNConv, coordinate-free, identical to Baseline v1.1).
   - Cross-attention block (4 heads, coordinate-free, identical to Baseline v1.1).
   - Regressor head (MLP, identical to Baseline v1.1).
   - Data preprocessing, splits, seeds, epoch count (40), batch size (8), optimizer (AdamW), scheduler (CosineAnnealingLR).
5. **Benchmark Configuration**:
   - Sub-dataset: Local subset ($N=1,000$).
   - Train/Val/Test Splits: Deterministic random splits (800/100/100) per seed.
   - Run commands: `venv\Scripts\python benchmark.py --model_type geometry ...`

---

## 2. Statistical Analysis & Outcome Summary

### 6. Aggregate Metrics (Mean $\pm$ Std)
- **Primary Metric (RMSE)**:
  - Baseline v1.1: $1.6048 \pm 0.0581$
  - Geometry Model: $1.5442 \pm 0.0818$
- **Secondary Metric (MAE)**:
  - Baseline v1.1: $1.2685 \pm 0.0337$
  - Geometry Model: $1.2139 \pm 0.0628$
- **Pearson Correlation ($r$)**:
  - Baseline v1.1: $0.5064 \pm 0.0541$
  - Geometry Model: $0.5470 \pm 0.0197$

### 7. Statistical Tests (Paired two-sample, $N=5$)
- **RMSE Paired t-test**: $t = -1.7876, \quad p = 0.1484$ (Not significant)
- **RMSE Wilcoxon Test**: $W = 3.0, \quad p = 0.3125$
- **MAE Paired t-test**: $t = -2.6603, \quad p = 0.0564$ (Borderline significant)

### 8. Effect Sizes
- **RMSE Cohen's d**: $-0.7995$ (Medium-to-Large effect size)
- **MAE Cohen's d**: $-1.1897$ (Large effect size)

---

## 3. Scientific Conclusions & Decision
9. **Scientific Conclusions**:
   - **Hypothesis $H_1$ (Geometry improves accuracy)**: Rejected for RMSE, as the 3.77% directional improvement is not statistically significant ($p = 0.1484 > 0.05$).
   - **Hypothesis $H_2$ (Geometry reduces regression-to-the-mean)**: Rejected. The residual-vs-true affinity correlation is $-0.7722$ (compared to $-0.75$ in the baseline) and the trendline slope is $-0.6542$ (compared to $-0.66$ in the baseline).
   - **Hypothesis $H_4$ (Geometry improves spatial attention correlation)**: Rejected. The cross-attention weights remain weakly correlated with physical distances ($r \le 0.16$ for almost all heads and PDBs).
10. **Benchmark Compatibility**: Fully compatible. No protocol deviations occurred.
11. **Promotion Decision**: **DO NOT PROMOTE**. Retain Baseline v1.1 as the canonical reference.
12. **Known Limitations**:
    - Small sample size ($N=5$) limits statistical power.
    - Coordinate-free protein representation and coordinate-free cross-attention blocks restrict the influence of explicit geometry.
13. **Recommended Phase 3.1B**: Introduce distance-biased cross-attention (where attention weights are explicitly biased by physical ligand-residue distances) to evaluate whether direct geometry injection in the interaction layer improves performance.
14. **Repository Status**: Certified Baseline v1.1 remains unchanged and active. The geometry model is logged and preserved in the codebase.
