# ProtLigGNN Benchmark Specification Contract v1.1

This document defines the official, immutable specification for the ProtLigGNN protein-ligand binding affinity benchmark. Every candidate architecture, representation-learning method, or geometric neural network must satisfy this contract to be compared against the baseline.

---

## 1. Specification Metadata & Versions

- **Specification Name:** ProtLigGNN Benchmark Protocol v1.1
- **Official Software Environment:** ProtLigGNN Environment v1.0 (defined in [OFFICIAL_ENVIRONMENT.md](file:///d:/ProtLigGnn/OFFICIAL_ENVIRONMENT.md))
- **Target Benchmark Version:**
  - **Optimized Baseline v1.1:** Official frozen baseline reference (GPU-accelerated; 1,000 samples, 5 seeds, 20 epochs, patience 5, Cosine scheduler).

---

## 2. Dataset Specification & Integrity

The benchmark is executed on a local subset of the PDBbind v2020 General Set. The files must match the following SHA-256 checksums to be considered valid:

### Index Files Checksums
- `index/INDEX_general_PL.2020R1.lst`: `35bacd30ce3731704f7adf60dff62df554ec9418f3f12d4411eb8baed3194b3d`
- `index/INDEX_general_NL.2020R1.lst`: `3e84feb77ad1d7323dde41c8c72bfe1ec2fca11c3d200c4749c6b1715a19071c`
- `index/INDEX_general_PN.2020R1.lst`: `42bd364b2da062a46184a86f3844ba29218c037393e9d5f50915b308be279c71`
- `index/INDEX_general_PP.2020R1.lst`: `5a42fb9c2f8c637e9fb35ad7cf145be5bc87ee98e74c277208faa95d788ecfe5`

---

## 3. Dataset Splitting Protocol

To prevent chemical data leakage and ensure reproducible splits, training/validation/testing IDs must be generated via the following protocol:

- **Split Strategy:** Deterministic Random Split (`deterministic_random_split`).
- **Split Fractions:**
  - **Train:** 80% of complexes.
  - **Validation:** 10% of complexes (used for early stopping).
  - **Test:** 10% of complexes (held out exclusively for final reporting).
- **Leakage Constraints:**
  - \( \text{Train} \cap \text{Val} = \emptyset \)
  - \( \text{Train} \cap \text{Test} = \emptyset \)
  - \( \text{Val} \cap \text{Test} = \emptyset \)
  - Overlaps are audited before training by [validate_split_manifest](file:///d:/ProtLigGnn/experiment_infra.py#L206) and must report zero violations.

---

## 4. Training & Hyperparameter Standards

All benchmark runs must be trained with the following hyperparameter contract:

| Parameter | Value | Rationale |
|---|---|---|
| **Batch Size** | `8` | Ensures low memory footprint and keeps standard gradient updates. |
| **Optimizer** | `AdamW` | Replaces standard Adam for better weight decay performance. |
| **Learning Rate (LR)** | `1.9e-4` | Optimized initial learning rate. |
| **Weight Decay** | `1.17e-4` | Regularization coefficient to prevent GNN overfitting. |
| **Dropout** | `0.3181` | Dropout rate applied to GAT and regressor layers. |
| **Hidden Dimension** | `128` | Balance between representation capacity and parameter efficiency. |
| **Patience** | `5` | Epoch threshold for early stopping based on validation RMSE. |
| **LR Scheduler** | `Cosine` | Smooth learning rate decay to fine-tune final convergence. |
| **Gradient Clipping** | `1.0` | Bounds gradient magnitudes to stabilize attention backpropagation. |
| **Random Seeds** | `[42, 123, 777, 2024, 3407]` | 5 seeds to generate statistical confidence bounds. |
| **Determinism Seeding** | Enabled | `CUBLAS_WORKSPACE_CONFIG=":4096:8"` must be set to enforce deterministic CUDA matrix multiplications. |

---

## 5. Evaluation Metrics & Statistical Methodology

Models must be evaluated on the held-out Test set.

### 1. Regression Metrics
- **RMSE (Root Mean Squared Error):** The primary benchmark metric.
- **MAE (Mean Absolute Error):** Measures absolute prediction errors.
- **PCC (Pearson Correlation Coefficient):** Measures linear correlation.
- **Spearman (Spearman Rank Correlation):** Measures monotonic rank correlation.
- **\(R^2\) (Coefficient of Determination):** Measures goodness of fit.

### 2. Statistical Significance Tests
When comparing a new model against the baseline:
- **Paired Two-Sample t-test:** Run across the 5 seed metrics to compute \(p\)-values. A change is considered statistically significant only if \(p < 0.05\).
- **Bootstrap 95% Confidence Intervals:** Computed over 1,000 iterations to measure error bounds on predictions.

---

## 6. Verification & Acceptance Criteria

A benchmark run is accepted into the registry ONLY if:
1. **Artifact Validation Passes:** The `validate_experiment_artifacts` report confirms the presence and checksum validation of all required run outputs (checkpoints, CSV history, predictions, config fingerprint, environment metadata).
2. **Disjointness Confirmed:** `split_validation.json` verifies zero split overlaps.
3. **Registry Entry Recorded:** A run row is appended to `experiments/registry.csv`.

---

## 7. Change Management Rules

To prevent benchmark drift, any deviation from this specification requires a **version promotion**:

- **Major Version Promotion (e.g. v1.1 -> v2.0):**
  - Triggered by changing the raw dataset version (e.g., migrating from PDBbind 2020 to PDBbind 2024), adding new atom/pocket features, or altering the definition of labels (pKd/pKi).
  - *Action:* Retrain all baselines and freeze a new major benchmark release.
- **Minor Version Promotion (e.g. v1.1 -> v1.2):**
  - Triggered by changing the split strategy (e.g., from Random to Scaffold or Temporal split), changing the initial hyperparameters, or adding new primary evaluation metrics.
  - *Action:* Re-run the benchmark suite across all seeds.
