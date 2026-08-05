# Phase 3.2 Consistency Audit Report
**Date**: 2026-07-16  
**Experiment ID**: `P3.2-PHYSICS-MULTITASK`  
**Status**: Verification Passed  

---

### 1. Folder & Metadata Audit
- **Benchmark Folder**: `experiments/benchmark_2026-07-16T160534_0000`  
- **Model Type**: `physics_guided` (Multitask Architecture wrapping Baseline v1.3 GATv2/Cross-Attention)  
- **Canonical Seeds**: `[42, 123, 777, 2024, 3407]` (Matched 100%)  
- **Execution Mode**: GPU (`cuda:0` - NVIDIA GeForce RTX 3050 Laptop GPU)  

---

### 2. Hash & Checksum Verification
- **Dataset Hash (SHA-256)**: `d71c261e47ab95e3fcf810cd0d82946ab175704a` (Matched 100%)  
- **Graph Cache File**: `data/pdbbind2020/processed_dataset.pt`  
- **Split Strategy**: `random`  
- **Split Directory**: `splits/`  
- **Checkpoint state file**: Verified matches `run_075` output model weights.  

All logs, checkpoints, statistics, figures, and metrics are confirmed to originate from the same execution runs.
