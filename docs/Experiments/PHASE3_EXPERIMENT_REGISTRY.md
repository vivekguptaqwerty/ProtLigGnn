# Phase 3 Controlled Experiment Registry
**Registry File**: Canonical Ledger for Phase 3 Research  
**Status**: Active  

This registry tracks all completed Phase 3 experiments, their parameters, performance deltas, and promotion outcomes.

---

## 1. Registry Table

| Experiment ID | Date | Model Type | Parameters Changed | Mean RMSE | PCC | p-value | Promotion Status |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **P3.1A-GEOMETRY-LIGAND** | 2026-07-12 | `geometry` | Ligand GAT edge-distance attributes | 1.5442 | 0.5470 | 0.1484 | **REJECTED (Retained v1.1)** |

---

## 2. Certified Reference Baselines

For reference, the certified baseline metrics are:

### Optimized Baseline v1.1
- **Mean RMSE**: $1.6048 \pm 0.0520$
- **Mean MAE**: $1.2685 \pm 0.0301$
- **Mean PCC**: $0.5064 \pm 0.0484$
- **Mean Spearman**: $0.4911 \pm 0.0405$
- **Mean $R^2$**: $0.2006 \pm 0.0791$
- **Total Parameters**: 362,881
- **Status**: **Certified Baseline Reference**
