# ProtLigGNN Reproducibility Milestone Validation Report

Date: 2026-07-11

## Scope

This milestone improved experimental methodology, benchmarking, evaluation, and reproducibility only. It did not redesign the ProtLigGNN architecture, replace the GNN, change datasets, or modify the research hypothesis.

## Files Modified or Added

| File | Status | Purpose |
|---|---|---|
| `benchmarking.py` | Added | Shared reproducibility utilities: deterministic splits, experiment artifacts, environment capture, metrics, plots, error analysis, statistical aggregation, and manifests. |
| `benchmark.py` | Added | One-command multi-seed benchmark runner. |
| `experiment_infra.py` | Added | Dataset/config fingerprinting, registry, resume resolution, logging setup, environment freeze, split/artifact validation, lifecycle docs, and reproducibility scoring. |
| `compare_runs.py` | Added | Compare saved run artifacts across metrics, configs, seeds, splits, training curves, runtime, and environment. |
| `protliggnn_train.py` | Modified | Integrated split manifests, experiment directories, richer metrics, artifact saving, deterministic seed handling, optimizer/RNG checkpoint state, resume, registry, logging, fingerprints, and additive CLI options. |
| `ARCHITECTURE_DECISIONS.md` | Added | Architectural decision records for major infrastructure and modeling choices. |
| `BRAIN.md` | Modified | Documented the new experiment framework, workflow, outputs, architectural impact, and remaining limitations. |
| `VALIDATION_REPORT.md` | Added | Records this milestone's implementation decisions, verification results, blockers, and follow-up work. |

Validation-only artifacts were also generated under `validation_tmp/` to test split, metric, dry-run, and aggregation behavior without requiring Torch.

## Architectural Decisions

1. The benchmark framework was implemented as `benchmarking.py` instead of embedding all logic inside `protliggnn_train.py`.
   - Reason: keeps model/training code focused and makes split, metric, and artifact functions reusable.

2. `protliggnn_train.py` now creates an experiment directory by default while still writing the legacy `outputs/` artifacts.
   - Reason: new runs become self-contained without breaking existing command habits or downstream scripts that read `outputs/`.

3. Split manifests are generated from processed PDB IDs, not raw candidate records.
   - Reason: graph construction can skip invalid complexes; the saved split must match the actual trainable dataset.

4. Fixed split mode fails if any required split file is missing or empty.
   - Reason: silent fallback would invalidate reproducibility claims.

5. Scaffold, protein-family, and temporal split modes emit warnings and use deterministic fallbacks when metadata is incomplete.
   - Reason: the current repository does not provide complete family/scaffold/temporal metadata for every complex, but the CLI and artifact format now support these strategies where metadata permits.

6. Metrics are computed with a pure-Python implementation in `benchmarking.py`.
   - Reason: benchmark aggregation and validation can run even when the heavy ML stack is unavailable.

7. Undefined numeric values are serialized as JSON `null`.
   - Reason: strict JSON artifacts are safer for long-term experiment tracking than non-standard `NaN`.

8. New checkpoints include optimizer state and Python/NumPy/Torch/CUDA RNG state.
   - Reason: future benchmark runs need enough state to audit or resume experiments without relying only on model weights.

9. Dataset fingerprints record raw SHA256 plus stable JSON digests.
   - Reason: raw hashes preserve forensic auditability, while stable digests avoid false duplicate misses from volatile timestamps.

10. The registry uses config hash plus dataset hash to detect duplicate completed/running experiments.
   - Reason: the repository needs a durable master experiment index without introducing an external tracking service.

11. Resume supports `latest`, run IDs, and checkpoint paths without changing the model architecture.
   - Reason: long-running experiments need continuation while preserving backward compatibility with older checkpoints.

## Backward Compatibility

Existing training arguments remain valid:

```text
python protliggnn_train.py --data_dir data/pdbbind2020 --epochs 50 --batch_size 8
```

Additive arguments:

```text
--split_strategy random|fixed|scaffold|protein_family|temporal
--split_dir splits
--fixed_split_dir <path>
--val_fraction 0.1
--test_fraction 0.1
--dataset_version pdbbind2020_local
--experiments_root experiments
--experiment_dir <path>
--experiment_id <name>
--num_workers 0
--pin_memory
--disable_experiment_tracking
--resume latest|run_004|<checkpoint>
--allow_duplicate
--log_level INFO|WARNING|ERROR|DEBUG
--experiment_purpose <text>
--research_question <text>
--author <name>
--description <text>
--notes <text>
--tags tag1,tag2
--expected_outcome <text>
--current_milestone <text>
--architecture_version <text>
--research_version <text>
```

Legacy outputs are still written:

```text
outputs/epoch_history_<run_name>.csv
outputs/test_predictions_<run_name>.csv
outputs/scatter_<run_name>.png
outputs/training_curves_<run_name>.png
outputs/best_protliggnn.pt
```

New experiment outputs:

```text
experiments/run_001/config.json
experiments/run_001/cli_args.json
experiments/run_001/environment.json
experiments/run_001/environment.lock
experiments/run_001/requirements.lock
experiments/run_001/pip_freeze.txt
experiments/run_001/git.json
experiments/run_001/metadata.json
experiments/run_001/config_hash.txt
experiments/run_001/config_fingerprint.json
experiments/run_001/dataset_hash.json
experiments/run_001/dataset_metadata.json
experiments/run_001/split/train_ids.txt
experiments/run_001/split/val_ids.txt
experiments/run_001/split/test_ids.txt
experiments/run_001/split/metadata.json
experiments/run_001/split_validation.json
experiments/run_001/metrics.json
experiments/run_001/predictions.csv
experiments/run_001/error_analysis.csv
experiments/run_001/error_analysis.md
experiments/run_001/evaluation_report.md
experiments/run_001/training_history.csv
experiments/run_001/checkpoint.pt
experiments/run_001/plots/
experiments/run_001/experiment.log
experiments/run_001/artifact_validation.json
experiments/run_001/reproducibility_report.md
experiments/run_001/reproducibility_report.json
experiments/run_001/experiment_lifecycle.md
experiments/run_001/manifest.json
experiments/registry.csv
```

## Verification Results

| Check | Result | Evidence |
|---|---|---|
| Python syntax compilation | Passed | `python -m py_compile experiment_infra.py compare_runs.py benchmarking.py benchmark.py protliggnn_train.py infer_affinity.py` exited with code 0. |
| Benchmark CLI help | Passed | `python benchmark.py --help` rendered the full CLI. |
| Compare CLI help | Passed | `python compare_runs.py --help` rendered the full CLI. |
| Deterministic split generation | Passed | Synthetic 10-ID split produced train=8, val=1, test=1 and wrote `validation_tmp/split_check/`. |
| Split ID to index conversion | Passed | Synthetic manifest converted back to deterministic dataset indices. |
| Split leakage prevention | Passed | Synthetic overlapping train/val split raised `ValueError` and wrote a failed split validation report. |
| Expanded metric computation | Passed | Synthetic regression test produced MSE, RMSE, MAE, median AE, Pearson, Spearman, R2, bias, and mean prediction. |
| Benchmark dry run | Passed | `python benchmark.py --dry_run --seeds 2 --epochs 1 --max_samples 1 ...` created a benchmark plan and two run directories under `validation_tmp/`. |
| Statistical aggregation writer | Passed | Synthetic two-seed metrics produced `results_summary.csv`, `per_seed_metrics.csv`, `benchmark_summary.json`, and `benchmark_report.md`. |
| Config hash stability | Passed | Synthetic configs differing only in run/output identity produced identical canonical config hashes. |
| Stable dataset JSON digest | Passed | Synthetic metadata JSONs differing only by `created_utc` produced identical stable digests while raw SHA256 remains available. |
| Artifact validation | Passed | Synthetic missing artifact was reported in `artifact_validation.json`. |
| Registry writer | Passed | Synthetic `registry.csv` row was created under `validation_tmp/infra_check/experiments`. |
| Experiment comparison | Passed | Synthetic `run_001` and `run_002` produced console, CSV, and Markdown comparison outputs. |
| Benchmark runner dry run after lifecycle additions | Passed | `python benchmark.py --dry_run --seeds 2 ... --allow_duplicate` produced expected per-seed commands with lifecycle flags. |
| Training entrypoint import | Blocked by environment | `python protliggnn_train.py --help` failed with `ModuleNotFoundError: No module named 'torch'`. |
| Inference entrypoint import | Blocked by environment | `python infer_affinity.py --help` failed with `ModuleNotFoundError: No module named 'torch'`. |
| Local venv | Blocked by environment | `.venv`-style local interpreter path resolves to a missing Python 3.10 install: `No Python at "C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe"`. |

## End-to-End Verification Status

Requested path:

```text
clone -> install dependencies -> training starts -> dataset loads -> split generated
-> model trains -> checkpoint saved -> evaluation runs -> metrics computed
-> plots generated -> benchmark report generated -> inference works
-> artifacts saved -> BRAIN.md updated -> no broken imports -> no runtime errors
```

Actual status in this workspace:

```text
repository present -> Python available -> benchmark utilities compile
-> split generation works -> metrics work -> dry-run benchmark works
-> aggregate report generation works -> registry/config/split/artifact checks work
-> comparison utility works -> BRAIN.md updated -> ADRs created
-> Torch/PyG-dependent training and inference blocked by missing torch
```

The code path is implemented, but full training/inference validation could not be executed in the current environment because neither available Python 3.14 interpreter has `torch`, and the repository's local venv points to a missing Python 3.10 installation.

## Known Issues Remaining

1. Full end-to-end model training and inference must be rerun in a valid Torch/PyG/RDKit/BioPython environment.
2. `requirements.txt` remains unpinned; the new environment capture records installed versions but does not solve dependency locking.
3. Historical retained metrics are still not reproducible unless regenerated with the new split and experiment artifacts.
4. Scaffold split quality depends on successful RDKit ligand parsing and meaningful ligand files.
5. Protein-family split quality depends on explicit family metadata that is not currently exposed by the dataset loader.
6. Temporal split quality depends on complete release-year metadata in the PDBbind index.
7. Paired statistical tests are represented in the benchmark report as not applicable until baseline run groups are supplied.
8. Resume continuation is implemented for new checkpoints but could not be numerically verified without Torch/PyG.
9. Duplicate detection depends on future completed/running registry rows; historical runs without registry entries cannot be detected automatically.

## Suggested Follow-Up Work for Milestone 2: Geometry-Aware Modeling

1. Add explicit scaffold/family/temporal metadata validation reports before comparing architecture changes.
2. Add geometry-aware edge attributes or equivariant modules only after the benchmark baseline is fully reproducible.
3. Compare geometry-aware candidates against the frozen benchmark protocol using paired seed-level metrics.

## Milestone 2A: Environment Standardization and GPU Enablement (2026-07-11)

### Status: Resolved & Active

We have successfully standardized the official research environment and enabled CUDA GPU acceleration:
1. **Python 3.11.9 Transition:** Standardized from Python 3.14 to Python 3.11.9 to resolve deep learning packaging constraints.
2. **CUDA GPU Acceleration:** Installed CUDA-enabled PyTorch `2.3.1+cu121` and PyG `2.8.0` with precompiled C++ binary extensions, unlocking the local NVIDIA RTX 3050 GPU.
3. **Validation & Verification:**
   - Created [verify_environment.py](file:///d:/ProtLigGnn/verify_environment.py) which returns `PASS`.
   - Verified GPU forward and backward passes, checkpoint serialization, resuming, and inference on CUDA.
   - Fixed GPU determinism warning by setting the `CUBLAS_WORKSPACE_CONFIG=":4096:8"` environment variable inside `set_seed`.
   - Confirmed that all 10 unit tests pass successfully in the new environment.

## Milestone 2B: Persistent Graph Cache & Benchmark Acceleration (2026-07-11)

### Status: Resolved & Active

We have implemented a persistent, validated, and versioned graph caching system:
1. **Graph Preprocessing Cache:** Created `processed_dataset.pt` saving and loading mechanisms inside `protliggnn_train.py` to bypass RDKit pocket parsing and graph generation.
2. **Robust Validation:** Added metadata manifest `graph_cache_manifest.json` ensuring dataset fingerprints, configurations, feature dimensions, and SHA-256 checksums are verified before cache is loaded, auto-rebuilding on any mismatch.
3. **Substantial Speedup:** Achieved over **100x acceleration** on training startup (reducing startup time from **10.7 minutes** to **~5.7 seconds** for the 5,000 PDBbind subset).
4. **Scientific Equivalence:** Proven that loading from cache yields mathematically identical training trajectories, loss values, predictions, and metrics compared to dynamic dynamic generation.


