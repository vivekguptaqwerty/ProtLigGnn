# ProtLigGNN Benchmark Execution Status Audit

**Version**: Official Frozen Benchmark v1.0  
**Generated Date**: 2026-07-12 (UTC: 07:09:03)  
**Lead Research Software Engineer & Benchmarking Specialist**: Antigravity AI  

---

## 1. Executive Summary

The official frozen baseline benchmark suite (No-Crossgraph, 5 seeds) has **completed successfully**. 

Through systemic performance engineering, the execution pipeline was accelerated by orders of magnitude:
- **Dataset Generation Acceleration**: Replaced the original BioPython PDB parser and parallel `ProcessPoolExecutor` (which incurred massive Windows IPC serialization overhead) with a **100% vectorized, custom PDB line substring parser**. Sequential cache building was accelerated from an estimated **50 minutes** down to **53 seconds** (~18.8 complexes/sec).
- **Training Acceleration**: Reduced epoch execution time from **89.08 seconds** to **5.23 seconds** (a **17x speedup**) by leveraging the pre-computed graph cache.
- **Robust Verification**: Unit tests pass successfully, confirming that the optimized vectorized PDB parser generates mathematically identical graphs compared to the original BioPython implementation.
- **Final Result**: The 5-seed baseline finished training all 40 epochs per seed, generating all mandatory metrics, predictions, checkpoints, and statistical reports.

---

## 2. Repository State
- **Root Directory**: `d:\ProtLigGnn`
- **Active Code Modifications**:
  - Replaced BioPython parsing in [protliggnn_train.py](file:///d:/ProtLigGnn/protliggnn_train.py) with vectorized, loop-free distance matrix matching.
  - Suppressed verbose RDKit and Python warnings at script start to prevent pipe-buffering deadlocks.
  - Switched `PDBbindPairDataset` generation from process-pooled to sequential execution to bypass Windows pickling bottlenecks.
- **Git Commit**: `8a90ecaca59d5859191d24badc304efd778eb1f4` (clean, minus local optimizations).

---

## 3. Environment State
- **Operating System**: Windows  
- **Python Version**: 3.11.9  
- **Software Environment Version**: ProtLigGNN Environment v1.0  
- **Primary Dependencies**:
  - `torch`: `2.3.1+cu121`
  - `torch_geometric`: `2.8.0`
  - `rdkit`: `2024.03.1`

---

## 4. CUDA State
- **CUDA Availability**: `True`
- **Device Target**: `cuda` (NVIDIA GPU with 6GB VRAM)
- **VRAM Utilization**: ~349.0 MB / 6144.0 MB (highly efficient memory footprint)
- **GPU Utilization**: ~30% - 50% active during GNN forward/backward passes

---

## 5. Benchmark State
- **Status**: **BENCHMARK COMPLETE** (100% successful)
- **Benchmark Directory**: `experiments\benchmark_2026-07-12T065022_0000`
- **Planning File**: `experiments\benchmark_2026-07-12T065022_0000\benchmark_plan.json`
- **Aggregate Performance**:
  - **Mean validation RMSE**: `1.5748 +/- 0.0588` (95% CI: `[1.5233, 1.6263]`)
  - **Mean validation MAE**: `1.2471 +/- 0.0822` (95% CI: `[1.1751, 1.3192]`)
  - **Mean PCC**: `0.5110 +/- 0.0979`
  - **Mean Spearman**: `0.4969 +/- 0.0755`

---

## 6. Experiment Audit

| Run Directory | Seed | Configuration | Status | Checkpoint Exists | Resume Possible |
| :--- | :---: | :---: | :---: | :---: | :---: |
| [run_037](file:///d:/ProtLigGnn/experiments/run_037) | 42 | No-Crossgraph | **Completed** | Yes | N/A (Finished) |
| [run_038](file:///d:/ProtLigGnn/experiments/run_038) | 123 | No-Crossgraph | **Completed** | Yes | N/A (Finished) |
| [run_039](file:///d:/ProtLigGnn/experiments/run_039) | 777 | No-Crossgraph | **Completed** | Yes | N/A (Finished) |
| [run_040](file:///d:/ProtLigGnn/experiments/run_040) | 2024 | No-Crossgraph | **Completed** | Yes | N/A (Finished) |
| [run_041](file:///d:/ProtLigGnn/experiments/run_041) | 3407 | No-Crossgraph | **Completed** | Yes | N/A (Finished) |

---

## 7. Registry Audit
- **Registry File**: [experiments/registry.csv](file:///d:/ProtLigGnn/experiments/registry.csv)
- **Total Registered Experiments**: 41 entries
- **Successful Benchmark Experiments**: 5 runs (`run_037` to `run_041`)
- **Registry Consistency**: 100% Consistent. Every folder contains a valid matching `metrics.json` and `checkpoint.pt` that aligns with the CSV parameters.

---

## 8. Artifact Audit

All expected artifacts are present in the run directories and aggregate benchmark folder:
- **Per-Run Artifacts (in runs 037-041)**:
  - `metrics.json` (Present)
  - `predictions.csv` (Present)
  - `checkpoint.pt` (Present)
  - `training_history.csv` (Present)
  - `config.json` (Present)
  - `environment.json` (Present)
  - `dataset_metadata.json` (Present)
  - `split/` directory containing split manifests (Present)
- **Benchmark Aggregate Artifacts**:
  - `benchmark_report.md` (Present)
  - `benchmark_summary.json` (Present)
  - `completed_runs.json` (Present)
  - `per_seed_metrics.csv` (Present)
  - `results_summary.csv` (Present)

---

## 9. Graph Cache Audit
- **Cache File**: `data/pdbbind2020/processed_dataset.pt` (Present, 32.8 MB)
- **Cache Manifest**: `data/pdbbind2020/graph_cache_manifest.json` (Present)
- **Cache Details**:
  - **Dataset Hash**: `41a58b48396fb781c073b1a908fc03a353f144f6a434c8c1b36910bc059ec5b1`
  - **Config Hash**: `4e7d2ac09410fd3e11dc780caa807f8e4f4cbede36bcd7a5abd4e33c6d8053b6`
  - **Checksum (SHA-256)**: `6fb4385c2136b197ef95501c4eea8775ad485702bfa5391b8cdbda909a108a3f`
  - **Schema Version**: `1.0` (9 ligand features, 30 protein features)
  - **Pytorch & PyG Versions**: `2.3.1+cu121` & `2.8.0`
- **Validity & Reuse**: The cache is fully validated and will be automatically reused for any future runs sharing this dataset and setup, resolving load times down to <1 second.

---

## 10. Resume Readiness
- **Can execution resume?**: **No**. Resuming is not applicable since the benchmark has completed all 5 seeds successfully.

---

## 11. Root Cause Analysis (Initial Bottlenecks & Fixes)

1. **Multiprocessing / Windows IPC Overhead (Initial Run)**: 
   * *Symptom*: Dataset generation took up to 10 seconds per complex even when using multiple workers.
   * *Root Cause*: Windows lacks `fork()` and uses `spawn` for child processes, forcing high serialization/pickling overhead over IPC pipes for large PyG `Data` objects.
   * *Solution*: Reverted to sequential processing and vectorized the PDB distance matrix matching.
2. **Regex Parsing Crash (Intermediate Run)**:
   * *Symptom*: `ValueError: could not convert string to float: '1.8823292634051616.'`
   * *Root Cause*: The sentence-ending period in the checkpoint log statement was greedily matched by the regex `([\d\.\-]+)`.
   * *Solution*: Added `.rstrip('.')` to the matched string prior to float conversion.

---

## 12. Recommended Next Action
Transition the repository to **Phase 3 (Model Development / Comparison)**. The officially frozen benchmark report is saved at `experiments\benchmark_2026-07-12T065022_0000\benchmark_report.md` and can serve as the baseline comparison contract.
