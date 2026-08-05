# Phase 3.1A Promotion Decision
**Experiment ID**: P3.1A-GEOMETRY-LIGAND  
**Decision Date**: 2026-07-12  
**Final Decision**: **REJECTED (Baseline Retained)**  

This document evaluates the promotion criteria for the geometry-aware model to decide whether to replace Certified Baseline v1.1.

---

## 1. Promotion Criteria Evaluation

| Criterion | Requirement | Current Outcome | Verification Status |
| :--- | :--- | :--- | :---: |
| **Performance Superiority** | Mean RMSE must improve compared to Baseline v1.1 ($1.6048$) | Mean RMSE is **$1.5442$** (Directional improvement of 3.77%) | **PASSED** |
| **Statistical Significance** | Paired t-test on RMSE across 5 seeds must yield $p < 0.05$ | Paired t-test yields **$p = 0.1484$** (Not significant) | **FAILED** |
| **Reproducibility Protection** | No regressions in baseline performance or execution | Verified via `test_baseline_equivalence.py` | **PASSED** |
| **Protocol Adherence** | No deviations from splits, seeds, or training parameters | Checked dataset hash and split ID manifests | **PASSED** |
| **Repository Integrity** | No changes to frozen files or benchmark layouts | Validation tests passed successfully | **PASSED** |

---

## 2. Rationale & Final Decision
- **Final Decision**: **DO NOT PROMOTE**. The candidate model `ProtLigGNNGeometry` is rejected for promotion to Baseline v2.0. **Optimized Baseline v1.1 remains the canonical reference baseline**.
- **Reasoning**: While the geometry model shows directional improvements across all metrics (3.77% in RMSE, 4.30% in MAE, 10.71% in Spearman correlation), we cannot rule out random initialization variance as the source of this difference because the paired t-test p-value ($0.1484$) is larger than the significance threshold ($0.05$). Under strict scientific methodology, we must treat this as a **statistically equivalent result**.

---

## 3. Consequences
- **Immutable Baseline**: Optimized Baseline v1.1 continues to serve as the active frozen reference for all future Phase 3 active research comparison runs.
- **Scientific Logging**: The geometry experiment runs (run_047 to run_051) are registered in the registry, and all code is retained in the repository under the `--model_type geometry` flag. This preserves the scientific audit trail of negative/equivalent results, which is essential for publication integrity.
