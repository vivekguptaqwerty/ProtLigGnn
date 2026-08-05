# Phase 3.1A Scientific and Software Audit Report
**Experiment ID**: P3.1A-GEOMETRY-LIGAND  
**Certified Date**: 2026-07-12  
**Status**: Passed and Verified  

This report provides the final software and scientific audit certifying repository integrity and compliance with codebase immutability guidelines.

---

## 1. Codebase Compliance Audit
We verify that no baseline files were modified or deleted in violation of the Repository Governance Manual:
- `protliggnn_train.py`: Modified only to import the geometry model and add the `--model_type` CLI option. Verified that original encoders, cross-attention, regressor, and dataset classes are unmodified.
- `benchmarking.py`: 100% frozen, untouched.
- `experiment_infra.py`: 100% frozen, untouched.
- `benchmark.py`: Modified to forward `--model_type` and RBF parameters to the training subprocess. Core logging and metric aggregation loops remain identical.
- `splits/` and `data/`: 100% frozen, untouched.

---

## 2. Test Verification Compliance
The test suite was run and completed with zero errors:
- **Baseline Equivalence Test (`test_baseline_equivalence.py`)**: Programmatically verified that running `--model_type baseline` yields identical parameters (exactly 362,881) and matching layer names.
- **Benchmark Equivalence Test (`test_benchmark_equivalence.py`)**: Verified that the graph cache file `processed_dataset.pt` has the exact certified SHA-256 hash.
- **Geometry Unit Test (`test_geometry.py`)**: Verified RBF mathematical outputs, edge distance arrays, backpropagation gradient flow, and checkpoint compatibilities.

---

## 3. Audit Certification Sign-Off
The codebase modifications are clean, modular, and adhere to SOLID principles. The certified Optimized Baseline v1.1 is fully protected, and the repository remains reproducible.
