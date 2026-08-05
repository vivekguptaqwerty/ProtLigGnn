# Phase 3.1A Benchmark Execution Plan
**Experiment ID**: P3.1A-GEOMETRY-LIGAND  
**Execution Type**: Multi-Seed Benchmark Evaluation  
**Status**: Ready for Execution  

This document defines the execution schedule, resource requirements, runtime estimations, and reporting protocols for the official benchmark evaluation of the geometry model.

---

## 1. Execution Schedule & Protocol
The benchmark will run the training and evaluation loop across the **5 canonical seeds** using the following commands:

```bash
# Seed 42
venv\Scripts\python protliggnn_train.py --model_type geometry --seed 42 --epochs 40 --device cuda --run_name run_p3_1a_s42

# Seed 123
venv\Scripts\python protliggnn_train.py --model_type geometry --seed 123 --epochs 40 --device cuda --run_name run_p3_1a_s123

# Seed 777
venv\Scripts\python protliggnn_train.py --model_type geometry --seed 777 --epochs 40 --device cuda --run_name run_p3_1a_s777

# Seed 2024
venv\Scripts\python protliggnn_train.py --model_type geometry --seed 2024 --epochs 40 --device cuda --run_name run_p3_1a_s2024

# Seed 3407
venv\Scripts\python protliggnn_train.py --model_type geometry --seed 3407 --epochs 40 --device cuda --run_name run_p3_1a_s3407
```

---

## 2. Resource & Runtime Estimates
- **Dataset Size**: 1,000 complexes (loaded from the pre-processed cache file `processed_dataset.pt`).
- **Throughput**: $\approx 190 \text{ complexes/second}$ (on CUDA GPU).
- **Epoch duration**: $\approx 5.3 \text{ seconds}$ per seed.
- **Estimated Epochs to Convergence**: Early stopping is typically triggered at $20 \to 25$ epochs.
- **Estimated Run Time Per Seed**: $\approx 2.5 \text{ minutes}$ (including dataloading and validation passes).
- **Total Benchmark Duration (5 Seeds)**: $\approx 12.5 \text{ minutes}$ on a single CUDA GPU.
- **Peak VRAM Memory**: $< 50 \text{ MB}$ (extremely lightweight).

---

## 3. Registry & Artifact Integration
After execution:
1. The experiment run folders will be written under `experiments/`.
2. Each run folder will contain:
   - `args.json`
   - `config_hash.txt`
   - `training_history.csv`
   - `best_model.pt`
   - `test_predictions.csv`
   - `evaluation_report.json`
3. The benchmark summary will be appended to the central registry `experiments/registry.csv`.
4. The Phase 3 Experiment Registry `PHASE3_EXPERIMENT_REGISTRY.md` will be updated with mean metrics and statistical tests.

---

## 4. Evaluation and Promotion Criteria
1. Calculate the mean and standard deviation of the Test RMSE across the 5 runs.
2. Run a paired t-test between the 5 `geometry` RMSE runs and the 5 certified `baseline` runs.
3. If mean RMSE is lower than $1.6048$ and the t-test $p < 0.05$, promote the geometry model to **Baseline v2.0**.
