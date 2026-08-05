# Phase 3.1A Readiness Certificate
**Experiment ID**: P3.1A-GEOMETRY-LIGAND  
**Certified Date**: 2026-07-12  
**Baseline Reference**: Optimized Baseline v1.1  
**Integrity Status**: Passed & Certified  

---

## 1. Executive Readiness Matrix

| Verification Vector | Status | Reference / Hash |
| :--- | :---: | :--- |
| **Dataset Integrity** | **PASSED** | Checked list of PDB files under splits |
| **Graph Cache Hash** | **VERIFIED** | `6fb4385c2136b197ef95501c4eea8775ad485702bfa5391b8cdbda909a108a3f` |
| **Baseline Parity** | **VERIFIED** | 362,881 parameter check |
| **Unit Verification** | **PASSED** | 8 passed tests (`test_geometry.py`) |
| **Protection Tests** | **PASSED** | 6 passed tests (`test_baseline_equivalence.py`, `test_benchmark_equivalence.py`) |
| **Smoke Validation** | **PASSED** | CPU & GPU convergence validated |

---

## 2. Risk Assessment & Mitigations
- **Overfitting Risk**: Adding explicit coordinate representations could increase overfitting on the small local dataset.
  - *Mitigation*: Dropout is fixed at $0.3181$ in the GATConv layers and the regressor head, matching the certified optimal hyperparameters.
- **Cache Mismatch**: Running local training with custom flags could taint the processed dataset cache.
  - *Mitigation*: The test suite programmatically asserts graph cache hashes before execution. A tainted cache is immediately caught by `tests/test_benchmark_equivalence.py`.

---

## 3. Platform Integrity Declaration
We certify that the ProtLigGNN repository has been successfully hardened and verified. The baseline configuration remains completely unaffected and reproducible.

---

## 4. Conclusion
Phase 3.1A has successfully introduced explicit ligand geometric edge representations while preserving full benchmark equivalence with the certified Optimized Baseline v1.1. The repository is scientifically reproducible and ready for controlled benchmarking.
