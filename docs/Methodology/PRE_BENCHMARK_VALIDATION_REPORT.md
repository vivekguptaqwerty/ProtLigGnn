# Pre-Benchmark Validation Report: Phase 3.1A Invariant Geometry
**Experiment ID**: P3.1A-GEOMETRY-LIGAND  
**Certified Date**: 2026-07-12  
**Status**: All Checks Passed — Authorized to Benchmark  

This report verifies the structural, environmental, and statistical integrity of the ProtLigGNN repository prior to running the official benchmark for the geometry-aware model.

---

## 1. Validation Matrix

| Parameter | Certified Reference Value / Requirement | Current Active Value | Verification Status |
| :--- | :--- | :--- | :---: |
| **Git Branch** | Target release branch | `main` | **PASSED** |
| **Repo Cleanliness** | All core code verified under git control | Modified only for CLI option & models directory | **PASSED** |
| **Baseline Parity** | `Sum(params) == 362,881` for baseline | Verified via `test_baseline_equivalence.py` | **PASSED** |
| **Dataset Hash** | `41a58b48396fb781c073b1a908fc03...` | Verified via `test_benchmark_equivalence.py` | **PASSED** |
| **Graph Cache SHA** | `6fb4385c2136b197ef95501c4eea8...` | Verified via `test_benchmark_equivalence.py` | **PASSED** |
| **Split Manifests** | Presence of `train_ids.txt` etc. | Checked split file sizes and directories | **PASSED** |
| **Seeds List** | `[42, 123, 777, 2024, 3407]` | Matched certified benchmark seeds | **PASSED** |
| **Registry File** | Presence of `experiments/registry.csv` | File validated and structured correctly | **PASSED** |
| **CUDA Available** | GPU enabled for training | NVIDIA CUDA 12.1 Enabled | **PASSED** |
| **PyTorch Version** | `2.3.1+cu121` | `2.3.1+cu121` | **PASSED** |
| **PyG Version** | `2.8.0` | `2.8.0` | **PASSED** |

---

## 2. Technical Findings & Environment Verification
- **CUDA Device**: Detected NVIDIA GPU. Double-precision operations, RNG seeding, and memory allocations are validated under standard CUDA 12.1 drivers.
- **RDKit Version**: `2024.03.1` (used for ligand conformer and covalent bond parsing during graph loading).
- **Test Suite Results**: Running `pytest tests/` executes 14 unit and protection tests with zero failures.

---

## 3. Benchmark Authorization
All integrity metrics match the certified reference specifications. The repository is scientifically locked and fully authorized to execute the Phase 3.1A benchmark.
