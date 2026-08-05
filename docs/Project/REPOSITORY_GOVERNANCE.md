# ProtLigGNN Repository Governance Manual

**Document Status**: Immutable  
**Certified Version**: v1.1-governed  
**Effective Date**: 2026-07-12  

This manual defines the operational principles, engineering standards, and scientific policies governing the ProtLigGNN repository. It serves as the constitutional guide for future developers, researchers, and maintainers.

---

## 1. Repository Core Principles

1. **Immutability of History**: Past experimental runs, registered benchmarks, and code commits representing baseline definitions must never be modified, deleted, or re-written.
2. **Reproducibility First**: Any model promoted to a baseline status must be fully reproducible from its registered configuration and environment lockfiles.
3. **Equitable Comparison**: All future architectures, representations, or geometric GNNs must be compared against the frozen baseline under identical evaluation parameters.
4. **Data Integrity**: The benchmark dataset and split protocols are frozen. Under no circumstances may data coordinates, splits, or labels be altered.

---

## 2. Versioning Policy

We apply semantic versioning separately to the Codebase and the ML Models:

### 2.1 Codebase Versioning (`vMAJOR.MINOR.PATCH`)
- **MAJOR**: Breaking API changes or dataset schema modifications (requires major milestone shift).
- **MINOR**: Backward-compatible feature additions (e.g., adding a new geometric encoder type in Phase 3).
- **PATCH**: Bug fixes or documentation updates that do not alter model outcomes.

### 2.2 Model & Baseline Versioning (`vMAJOR.MINOR`)
- **MAJOR**: Structural architectural revisions or changes to the target objective (e.g., v1.0 is the GCN/GAT/Cross-Attention regressor).
- **MINOR**: Parameter optimizations or hyperparameter refinements (e.g., Optimized Baseline v1.1).

---

## 3. Benchmark Policy & Governance

1. **Canonical Seeds**: All benchmarks must evaluate across the 5 official seed values: `[42, 123, 777, 2024, 3407]`. Single-seed reporting is prohibited for publications.
2. **Evaluation Directory**: The directory `experiments/benchmark_2026-07-12T081713_0000` is the canonical reference directory for Baseline v1.1.
3. **Metrics Consistency**: Primary evaluation metrics are Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Pearson Correlation Coefficient (PCC), Spearman Rank Correlation Coefficient, and Coefficient of Determination ($R^2$).

---

## 4. Dataset & Graph Cache Policy

1. **Local Dataset**: The benchmark utilizes PDBbind v2020 local subset (`max_samples=1000`).
2. **Pre-processing Immutability**: Node and edge feature dimension mappings (Ligand: 78, Protein: 30) are fixed.
3. **Graph Cache Manifest**: `graph_cache_manifest.json` tracks dataset hashes. Any change to the source files or parameters triggers a verification error. The processed cache file `data/pdbbind2020/processed_dataset.pt` is signed via SHA-256.

---

## 5. Experiment Promotion Policy

For any new model to be promoted to "Baseline Status", it must pass the following promotion gate:
1. **Performance Superiority**: Must exhibit a statistically significant improvement in RMSE over the current active baseline (p-value $< 0.05$ on a paired t-test across the 5 seeds).
2. **Parameter-Efficiency**: Model capacity must be documented. If parameter count is $>2\text{x}$ larger, the improvement must justify the computational overhead.
3. **Validation Suite Check**: Must pass the execution validation script (`validate_repository.py`) with zero errors.
4. **Environment Audit**: Must run successfully within the frozen Anaconda environment.

---

## 6. Bug-Fix and Patch Policy

1. **Allowed Edits**: Edits to the codebase during Phase 2 freeze are restricted to:
   - Genuine syntax bug fixes that prevent execution.
   - Documentation updates or formatting.
   - Code comments.
2. **Forbidden Edits**:
   - Modifying model architecture weights initialization.
   - Modifying loss functions.
   - Modifying test datasets or data loaders.

---

## 7. Review Checklist for Future Maintainers

Before merging any pull request in Phase 3, reviewers must verify:
- [ ] No changes have been made to files declared frozen in `REPOSITORY_FREEZE_REPORT.md`.
- [ ] The new experiment has been recorded in `experiments/registry.csv` with status `SUCCESS`.
- [ ] The PR includes a reproducibility report with a verification hash.
- [ ] Documentation (`BRAIN.md`) has been updated to reflect the new feature without altering baseline specifications.
