# ProtLigGNN Project Status Dashboard

**Last Hardening Date**: 2026-07-12  
**Current Code Version**: `v1.1.0-patch`  
**Current Model Version**: `Optimized Baseline v1.1`  
**Repository Maturity**: **Certified Research Platform (Phase 2 Locked)**  

---

## 1. Executive Status Matrix

| Component | Status | Verification Reference / Hash |
| :--- | :---: | :--- |
| **Official Dataset** | **FROZEN** | `41a58b48396fb781c073b1a908fc03a353f144f6a434c8c1b36910bc059ec5b1` |
| **Graph Cache** | **SIGNED** | `6fb4385c2136b197ef95501c4eea8775ad485702bfa5391b8cdbda909a108a3f` |
| **Benchmark Protocol**| **FROZEN** | 5 Seeds `[42, 123, 777, 2024, 3407]`, 10/10/80 Random Splits |
| **Python Environment**| **LOCKED** | `ProtLigGNN Environment v1.0` (PyTorch `2.3.1+cu121`) |
| **Registry Lineage** | **LOCKED** | `experiments/registry.csv` |
| **Baseline Promotion**| **PASSED** | Certified mean RMSE: **1.6048** (promoted v1.1) |

---

## 2. Certified Performance Summary

Evaluating the **Optimized Baseline v1.1** model over the 5 official seeds yields:
- **Mean RMSE**: $1.6048 \pm 0.0520$
- **Mean MAE**: $1.2685 \pm 0.0301$
- **Mean Pearson Correlation ($r$)**: $0.5064 \pm 0.0484$
- **Mean Spearman Rank Correlation**: $0.4911 \pm 0.0405$
- **Mean Coefficient of Determination ($R^2$)**: $0.2006 \pm 0.0791$

---

## 3. High-Priority Bottlenecks & Limitations

1. **Regression-to-the-Mean**: The model severely over-predicts low affinities ($+1.64$ error) and under-predicts high affinities ($-1.72$ error). This restricts screening selectivity.
2. **Implicit Spatial Representation**: The model lacks explicit 3D distance features in attention mechanisms, resulting in poor correlation ($r \approx 0.10$) with physical contact distances.
3. **Capacity Constraints**: The current lightweight model (1.4M params) underfits the PDBbind complex structure distribution, yielding a low overall $R^2$ of $\approx 0.20$.

---

## 4. Phase 3 Research Readiness

- [x] Pre-processed graphs cache verified and sealed.
- [x] Reproducibility verification suite running with zero errors.
- [x] Multi-seed ablation studies completed and documented.
- [x] Governance and repository freeze guidelines enacted.
- [x] Phase 3.1A (Geometry Feature Injection - Ligand-Only) successfully benchmarked, audited, and certified as statistically equivalent to Baseline v1.1.

**Current Verdict**: **PHASE 3.1A COMPLETED — CERTIFIED EQUIVALENT (Baseline Retained).**  
*All future models must compare against `Optimized Baseline v1.1` under identical seed splits and environment conditions.*
