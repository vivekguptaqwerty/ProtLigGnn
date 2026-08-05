# ProtLigGNN Repository Freeze Report

**Freeze Reference**: `PLGNN-FREEZE-20260712-V1.1`  
**Freeze Timestamp**: 2026-07-12T21:56:42Z (Local: 2026-07-12)  
**Certified Commit**: `8a90ecaca59d5859191d24badc304efd778eb1f4`  
**Certified Branch**: `main`  
**Freeze Level**: High (Immutable Baseline Platform)  

---

## 1. Scope of Freeze

To guarantee the validity of scientific comparisons in future publications, the following files and directories are frozen. No modification is allowed without review and approval by the Benchmark Committee.

### 1.1 Frozen Code & Pipelines
- [protliggnn_train.py](file:///d:/ProtLigGnn/protliggnn_train.py) — Core training, encoders, cross-attention, regressor, and loss functions.
- [benchmarking.py](file:///d:/ProtLigGnn/benchmarking.py) — Benchmark execution loops and reporting.
- [experiment_infra.py](file:///d:/ProtLigGnn/experiment_infra.py) — Registry management, hashes, lockfiles, and metrics extraction.
- [benchmark.py](file:///d:/ProtLigGnn/benchmark.py) — Entry point for canonical baseline runs.

### 1.2 Frozen Data & pre-processed Cache
- [data/pdbbind2020/processed_dataset.pt](file:///d:/ProtLigGnn/data/pdbbind2020/processed_dataset.pt) — Processed graph representations.
- [data/pdbbind2020/graph_cache_manifest.json](file:///d:/ProtLigGnn/data/pdbbind2020/graph_cache_manifest.json) — Manifest verification file.
- [splits/](file:///d:/ProtLigGnn/splits) — Test, train, and validation partition PDB lists.

### 1.3 Frozen Registry & Lineage
- [experiments/registry.csv](file:///d:/ProtLigGnn/experiments/registry.csv) — Primary audit trail of experimental history.
- [experiments/run_042/](file:///d:/ProtLigGnn/experiments/run_042) to [experiments/run_046/](file:///d:/ProtLigGnn/experiments/run_046) — The 5 official baseline runs (Seed 42, 123, 777, 2024, 3407).

---

## 2. Allowed vs. Forbidden Modifications

| Category | Allowed Modifications | Forbidden Modifications |
| :--- | :--- | :--- |
| **Model Code** | None | Changes to GCN/GAT/Cross-attention dim, activation, dropout, regression depth. |
| **Data Loader** | None | Changing sample size ($N=1000$), split ratios ($80/10/10$), or seed mappings. |
| **Environment** | None | Upgrading Python, PyTorch, PyG, or RDKit libraries outside official v1.0. |
| **Diagnostics** | Adding visualization scripts, logging tweaks, or non-intrusive analysis code. | Changing existing evaluation scripts or altering the metric definitions. |
| **Bug Fixes** | Syntax correction for interpreter crashes (requires patch version bump). | Logical edits that change model parameters or output predictions. |

---

## 3. Phase 3 Branching & Development Strategy

To start Phase 3 architecture research (such as equivariant GNNs, graph transformers, or geometric models):
1. **Branching**: Branch off from `main` at commit `8a90ecaca59d5859191d24badc304efd778eb1f4` with a descriptive name, e.g., `phase3/equivariant-gnn` or `phase3/graph-transformer`.
2. **Implementation**: Build the new architecture in a new module file, e.g., `models/equivariant.py`. Modify `protliggnn_train.py` ONLY to add support for the new model type under a CLI flag (e.g., `--model_type equivariant`).
3. **Execution**: Run the benchmark suite on the new model across the 5 seeds and record them in the registry under the new model version.
4. **Validation**: Run `validate_repository.py` to ensure zero regressions or breaks in baseline reproducibility.
5. **Promotion Review**: Request a review. If the model passes the promotion gates, it may be merged into `main` and registered as **Baseline v2.0**.
