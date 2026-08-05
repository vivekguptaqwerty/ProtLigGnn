# Phase 2 Plan Review: Hardware-Aware Executable Baseline & Benchmark Strategy

This document reviews and revises the Phase 2 plan to accommodate the discovered hardware capabilities, interpreter environment constraints, and available resources.

## Executive Summary

We performed a deep hardware and ecosystem audit on the host machine. We discovered that while the machine possesses an NVIDIA RTX 3050 GPU (6 GB VRAM), the virtual environment runs **Python 3.14.3**, for which no CUDA-enabled PyTorch wheels are currently compiled for Windows. Consequently, PyTorch execution is restricted to the CPU backend (`2.13.0+cpu`). 

Given the CPU constraint and limited available RAM (~3.88 GB free), running the full multi-seed benchmark on 5,000 samples for 40 epochs would take over **10 hours**, creating a performance risk. We recommend executing a **Validated Baseline v0.95** with 1,000 samples and 40 epochs across 5 seeds, which reduces runtime to a manageable **2 hours** while fully preserving statistical integrity.

---

## Current Status

### Repository State
- Core training logic is fully restored in [protliggnn_train.py](file:///d:/ProtLigGnn/protliggnn_train.py).
- Dataset affinity parser is corrected to match all 12,235 complexes.
- 10/10 tests are passing in [test_pipeline.py](file:///d:/ProtLigGnn/tests/test_pipeline.py).
- End-to-end training, single-pair inference, and resume functionalities have been verified.

### Infrastructure State
- An experiment registry (`experiments/registry.csv`) and run directory structure is active.
- Detailed metrics, predictions, plots, reproducibility reports, and lifecycle logs are generated automatically for each run.

### Benchmark State
- A dry run plan builder is active in [benchmark.py](file:///d:/ProtLigGnn/benchmark.py).
- The framework is now updated to load defaults from [benchmark_config.yaml](file:///d:/ProtLigGnn/benchmark_config.yaml).

---

## Hardware Analysis

- **Classification:** `CPU ONLY` (for PyTorch execution due to Python 3.14 limitations).
- **CPU:** Intel Core (8 physical cores, 12 threads).
- **RAM:** 15.73 GB total, ~3.88 GB free.
- **Acceleration Backend:** CPU fallback (`PyTorch 2.13.0+cpu`).

---

## Execution Strategy

1. **Sequential Runs:** To avoid memory thrashing and CPU over-subscription, runs will execute sequentially rather than in parallel.
2. **Resource Constraints:** Workers are set to `0` (`--num_workers 0`) and pinning is disabled (`--pin_memory false`) to keep memory footprint under 2 GB per run.
3. **Data/Epoch Scaling:** Cap training at 1,000 samples (`--max_samples 1000`) and 40 epochs (`--epochs 40`) per run.

---

## Risks & Mitigation

### Scientific Risks
- **Risk:** Capping dataset size at 1,000 samples (from 5,000) reduces representation of chemical scaffolds and protein families.
- **Mitigation:** The split remains deterministic and identical between the baseline and the model runs, ensuring a mathematically fair, controlled comparative study.

### Engineering & Performance Risks
- **Risk:** High memory usage leading to OOM or disk thrashing.
- **Mitigation:** Force single-threaded data loading (`--num_workers 0`) to prevent spawned process duplication in Windows.

---

## Recommended Benchmark Version

We recommend targeting **Validated Baseline v0.95** for this machine's run, as a CUDA-accelerated **Official Frozen Benchmark v1.0** is mathematically identical but computationally impractical on CPU within developer turn-around limits.

---

## Configuration Changes
- Added [benchmark_config.yaml](file:///d:/ProtLigGnn/benchmark_config.yaml) to centralize hyperparameters.
- Updated [benchmark.py](file:///d:/ProtLigGnn/benchmark.py) to read default values from the YAML configuration.

---

## Validation Strategy
- Validation will use the automated split disjointness check prior to training.
- Post-training validation will inspect the presence and checksum validation of all required artifacts (checkpoints, metrics, plots, prediction outputs).
