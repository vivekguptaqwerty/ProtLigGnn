# Phase 2.9 — Baseline Optimization, Ablation & Scientific Validation Implementation Plan v2.1

**Document Version:** v2.1 (Supersedes v2.0)  
**Status:** Under Final Review (Pending User Approval)  
**Lead Research Scientist & ML Architect:** Antigravity AI  

---

## 1. Executive Summary

This document specifies the finalized, publication-quality implementation plan for Phase 2.9 (Baseline Optimization, Ablation & Scientific Validation) of the ProtLigGNN repository. The primary objective is to maximize the performance of the existing ProtLigGNN GNN architecture on the local subset of the PDBbind v2020 General Set, without changing the scientific hypothesis, dataset splits, or feature definitions. 

This version introduces **formal baseline versioning (v1.0 -> v1.1)**, **staged evaluation to manage GPU compute budgets**, and a **multi-factor baseline governance framework**. These rules guarantee that historical baseline checkpoints, certificates, and metrics are preserved immutably while candidate baselines undergo strict peer-review style promotion gates before adoption.

---

## 2. Objectives

1. **Optimize Existing Architecture Capacity**: Locate the optimal hyperparameters (learning rate, weight decay, dropout, hidden dimension) using Bayesian search.
2. **Isolate Engineering vs. Scientific Effects**: Disentangle hyperparameter tuning/optimizations (AdamW, learning rate schedulers, gradient clipping) from architectural ablations (cross-attention, LayerNorm, residuals).
3. **Establish Rigorous Statistical Significance**: Determine if any optimized model statistically outperforms the frozen baseline ($p < 0.05$ via paired t-test, confidence intervals, and Cohen's $d$ effect size).
4. **Quantify Compute & Model Capacity**: Audit parameter count, memory footprint, throughput, and training dynamics to characterize the baseline's efficiency.
5. **Identify Systemic Failure Modes**: Perform an expanded post-prediction error analysis on physical/graph properties (affinity range, graph size, ligand/protein sizes) to find systematic model weaknesses.
6. **Implement Strict Benchmark Governance**: Ensure that all historical benchmarks are preserved under explicit versioned lineage identifiers, making it impossible to overwrite past baseline checkpoints or reports.

---

## 3. Assumptions & Constraints

- **Dataset Lock**: The dataset remains locked to the 1,000 PDBbind complexes defined in the Validated Baseline v0.95.
- **Split Lock**: Splits are fixed to the deterministic random train/val/test splits generated under seed `42` in [experiments/run_037/split](file:///d:/ProtLigGnn/experiments/run_037/split) (val_fraction=0.1, test_fraction=0.1).
- **Metric Lock**: The primary evaluation metric is validation/test Root Mean Squared Error (RMSE). Secondary metrics are MAE, Pearson ($r$), Spearman ($\rho$), and $R^2$.
- **Graph Pipeline Lock**: Graph node and edge features, pocket radius ($6.0\text{ \AA}$), and edge thresholds ($8.0\text{ \AA}$) are completely frozen.
- **Environment Lock**: Executions must run inside the official `ProtLigGNN Environment v1.0` using PyTorch `2.3.1+cu121` on the target NVIDIA GPU.

---

## 4. Experiment Budget Management (Staged Evaluation)

To maximize GPU efficiency and minimize carbon footprint while retaining absolute scientific rigor, we define a **3-stage evaluation pipeline**:

```
+------------------------------------+
| STAGE 1: Bayesian Search (Optuna)  |
| - 50 trials using Seed 42 only.    |
| - SQLite tracked to enable resume.  |
+------------------+-----------------+
                   |
                   v
+------------------------------------+
| STAGE 2: Candidate Ranking         |
| - Filter and rank HPO trials.      |
| - Select top 3 finalists.          |
+------------------+-----------------+
                   |
                   v
+------------------------------------+
| STAGE 3: Finalist Evaluation       |
| - Train top 3 candidates on all    |
|   5 seeds.                         |
| - Perform statistical tests.       |
+------------------------------------+
```

- **Stage 1: Bayesian Optimization (Seed 42 Only)**: Run a 50-trial Optuna Bayesian search targetting validation RMSE. Limiting to seed 42 reduces the initialization HPO footprint by 80%.
- **Stage 2: Candidate Ranking & Selection**: Programmatically sort and evaluate completed trials. Identify the **top 3 candidates** (based on lowest validation RMSE).
- **Stage 3: Full 5-Seed Finalist Evaluation**: Evaluate only these 3 finalists across the 5 benchmark seeds (`[42, 123, 777, 2024, 3407]`). This limits full seed training to high-probability configurations, saving significant GPU execution hours.

---

## 5. Detailed Modules

### Module 1: Architecture Audit
- **Goal**: Formally inspect parameter flow, LayerNorm positioning, activation functions, and layer sizing.
- **Implementation**:
  - Audit parameter counts per encoder (Ligand vs Protein) and regressor.
  - Assess gradient flow using PyTorch hooks to detect vanishing/exploding gradients.
  - **Deliverable**: `ARCHITECTURE_AUDIT.md` (ranking potential bottlenecks and improvements).

### Module 2: Training Diagnostics
- **Goal**: Evaluate baseline learning behavior (learning rate sensitivity, gradient norms, activation distributions).
- **Implementation**:
  - Run diagnostic trials to log training/validation loss curves on seed 42.
  - Measure the mean and variance of gradients across epochs.
  - **Deliverable**: Sections in `TRAINING_OPTIMIZATION_REPORT.md`.

### Module 3: Hyperparameter Optimization
- **Goal**: Perform the Stage 1 Bayesian optimization search using Optuna over seed 42.
- **Search Space**:
  - Learning Rate: Log-uniform $[10^{-4}, 3 \times 10^{-3}]$
  - Weight Decay: Log-uniform $[10^{-6}, 10^{-3}]$
  - Dropout: Uniform $[0.0, 0.4]$
  - Hidden Dimension: Choice $\{128, 256, 512\}$
- **Budget**: $50$ trials, optimizing validation RMSE.
- **Resumability**: Optuna trials will use an SQLite backend database (`experiments/optuna_study.db`).
- **Deliverable**: `HYPERPARAMETER_STUDY.md` (containing HPO trial metrics, parameter importance plots, and selection of the top 3 finalists).

### Module 4: Engineering Optimizations
- **Goal**: Evaluate performance gains from training tricks without modifying the GNN architecture.
- **Optimization Candidates**:
  - **Optimizers**: AdamW vs. RAdam vs. baseline Adam.
  - **Schedulers**: Cosine Annealing, ReduceLROnPlateau, and OneCycleLR.
  - **Gradient Clipping**: Grad norm thresholds of $\{1.0, 5.0, 10.0\}$.
  - **Warmup**: Linear warmup for the first $3$ epochs.
- **Deliverable**: `TRAINING_OPTIMIZATION_REPORT.md` (tabulating RMSE/MAE gains for each technique).

### Module 5: Scientific Ablation Studies
- **Goal**: Quantify the independent contribution of each architectural component.
- **Ablation Matrix** (varying one factor at a time):
  1. **No Crossgraph**: Bypassing Bidirectional Cross-Attention (replacing with zero-padded identity mapping).
  2. **No Attention**: Bypassing the MultiheadAttention layer, using plain global mean pooling of GNN outputs.
  3. **No LayerNorm**: Disabling all `LayerNorm` layers in the Encoders and Cross-Attention module.
  4. **No Residual**: Disabling residual connections inside `BidirectionalCrossAttention`.
  5. **Pooling Strategy**: Comparing Global Mean Pooling vs. Global Max Pooling.
- **Deliverable**: `ABLATION_STUDY.md` (tabulating validation and test metrics for each ablated candidate).

### Module 6: Error Analysis
- **Goal**: Expand prediction error audit to identify systematic model failures.
- **Analysis Variables**:
  - Correlation between prediction error (residuals) and true affinity range ($[0.0, 15.0]$).
  - Residuals vs. ligand size (number of heavy atoms).
  - Residuals vs. protein size (number of pocket residues).
  - Outlier identification (complexes with residuals $> 3.0$ RMSE).
- **Deliverable**: `ERROR_ANALYSIS_REPORT.md` (including scatter plots, residual distributions, and physical pocket analysis of failed cases).

### Module 7: Model Interpretability
- **Goal**: Extract attention weights and saliency maps to identify binding determinants.
- **Implementation**:
  - Extract attention maps from `BidirectionalCrossAttention` MultiheadAttention layers for 3 best-predicted and 3 worst-predicted complexes.
  - Compute gradient-based input saliency maps for protein pocket residues to identify key binding nodes.
  - **Deliverable**: `MODEL_INTERPRETABILITY_REPORT.md` (visualizing cross-attention weight distributions over target pocket residues).

### Module 8: Learning Dynamics
- **Goal**: Evaluate convergence speed and generalization gap.
- **Implementation**:
  - Analyze training vs. validation curves to identify the epoch where overfitting begins.
  - Compare generalization gaps (Train RMSE - Val RMSE) across different regularization regimes.
  - **Deliverable**: `LEARNING_DYNAMICS_REPORT.md`.

### Module 9: Statistical Validation
- **Goal**: Perform the Stage 3 validation of the finalists over all 5 seeds.
- **Implementation**:
  - Train the top 3 candidate finalists using seeds `[42, 123, 777, 2024, 3407]`.
  - Calculate paired t-test $p$-value, 95% bootstrap confidence intervals, and Cohen's $d$ effect size compared to the frozen baseline.
  - **Deliverable**: Detailed validation report embedded in `OPTIMIZED_BASELINE.md`.

### Module 10: Model Capacity & Computational Efficiency
- **Goal**: Profile memory footprint and execution latency of the optimized candidate.
- **Metrics**:
  - Parameter count and disk size.
  - GPU VRAM peak usage.
  - Inference latency per complex (milliseconds).
  - Benchmark throughput (complexes/second).
  - **Deliverable**: `MODEL_CAPACITY_REPORT.md`.

### Module 11: Baseline Promotion Decision
- **Goal**: Formally evaluate the optimized candidate baseline against the frozen baseline.
- **Promotion Gate Rules**:
  - The optimized model becomes the new official baseline if and only if:
    1. Validation/test RMSE improves with statistical significance ($p < 0.05$ via paired t-test).
    2. Model capacity remains within hardware limits (peak VRAM $< 1.0\text{ GB}$).
    3. Reproducibility checks are 100% successful.
  - Otherwise, the current frozen baseline is retained.
- **Deliverable**: `OPTIMIZED_BASELINE.md`.

### Module 12: Documentation & Repository Update
- **Goal**: Keep all project files synchronized with the results of Phase 2.9.
- **Implementation**:
  - Update `BRAIN.md`, `ROADMAP.md`, and `VALIDATION_REPORT.md` with final study findings.
  - **Deliverable**: `PHASE2_9_COMPLETION_REPORT.md`.

---

## 6. Baseline Versioning & Lineage Policy

To maintain a strict historical record of all baselines and prevent drift, the repository enforces a formal **Baseline Versioning Protocol**:

1. **Versioning Scheme**:
   - **Baseline v1.0**: The certified frozen baseline benchmark completed under seeds `[run_037 - run_041]`.
   - **Baseline v1.1**: The first optimized baseline configuration, should it pass the Module 11 promotion gate.
2. **Immutable Artifact Preservation**:
   - Under no circumstances will a historical checkpoint or report be overwritten.
   - Run directories from completed baselines are locked.
   - When a new baseline is promoted, it is assigned a **new version number**, a **new benchmark directory** under `experiments/benchmark_<timestamp>_v<version>`, a **new entry** in `experiments/registry.csv`, and a **new benchmark certificate JSON**.
   - The historical benchmark records remain in git and on the filesystem, enabling reproduction of older baselines at any time.

---

## 7. Benchmark Governance

The division between benchmark stability and experimental modeling is governed by the following rules:

### 7.1 Immutable Components
- **Dataset (Raw & Cached)**: The raw pdbbind general subset and the serialized `processed_dataset.pt` cannot be modified.
- **Dataset Splits**: The train/val/test partitions for each of the 5 seeds must remain bitwise identical across all versioned baselines.
- **Metrics Calculations**: Formulae and implementations of primary and secondary metrics must not change.

### 7.2 Versioned Components
- **Model Architecture Parameters**: Layer widths, attention heads, activation types, and normalization schemes.
- **Training Configuration**: Optimizers, learning rate, weight decay, dropout, learning rate schedulers, and epoch limits.

### 7.3 Deprecation Policy
- Old baseline models are marked as `DEPRECATED` in the global registry, but their configuration files and checkpoints are retained.
- Any publication draft or comparison table must reference both the initial baseline version (v1.0) and the promoted baseline version (v1.1) to show optimization lineage.

---

## 8. Final Success Criteria & Promotion Gate

The candidate model will be promoted to the new **Official Baseline (v1.1)** if and only if it satisfies all of the following:

- [ ] **Lower Mean RMSE**: The mean test RMSE across the 5 seeds is lower than that of Baseline v1.0 ($< 1.5748$).
- [ ] **Statistical Significance**: The improvement is statistically significant ($p < 0.05$ via paired t-test) OR practically meaningful with a large effect size (Cohen's $d > 0.8$).
- [ ] **Tightened Confidence Intervals**: 95% bootstrap confidence intervals for the candidate show a shift towards lower error compared to the baseline's confidence intervals.
- [ ] **Stable Seed Variance**: Standard deviation of test RMSE across the 5 seeds does not increase by more than $15\%$ compared to the baseline ($< 0.0676$).
- [ ] **Bitwise Reproducibility**: Running the configuration twice on seed 42 produces identical weights.
- [ ] **Resource Cost Acceptability**: Peak VRAM remains under $1.0\text{ GB}$, and training time per epoch remains under $10.0$ seconds on the target GPU.
- [ ] **Full Lineage Recording**: A new version tag (`v1.1`), new benchmark report, and new registry entries are generated without overwriting any files from Baseline v1.0.

If these criteria are met, the optimized baseline is promoted. If they are not met, the current baseline (v1.0) is retained as the active baseline, and Phase 2.9 results are frozen as a scientific negative result.

---

## 9. Expected Deliverables

The following artifacts will be written to the Brain artifacts directory:
1. `ARCHITECTURE_AUDIT.md`
2. `HYPERPARAMETER_STUDY.md`
3. `TRAINING_OPTIMIZATION_REPORT.md`
4. `ABLATION_STUDY.md`
5. `ERROR_ANALYSIS_REPORT.md`
6. `MODEL_INTERPRETABILITY_REPORT.md`
7. `LEARNING_DYNAMICS_REPORT.md`
8. `MODEL_CAPACITY_REPORT.md`
9. `OPTIMIZED_BASELINE.md`
10. `PHASE2_9_COMPLETION_REPORT.md`

---

## 10. Estimated Runtime

- **Stage 1 (Optuna HPO Study, 50 trials, Seed 42)**: $50 \text{ trials} \times \approx 30 \text{ sec/trial} \approx 25 \text{ minutes}$.
- **Stage 2 (Finalist Selection)**: Immediate ($< 2 \text{ seconds}$).
- **Stage 3 (Finalist Seed Evaluations, 3 candidates $\times$ 5 seeds)**: $15 \text{ runs} \times \approx 4 \text{ minutes/run} \approx 60 \text{ minutes}$.
- **Ablation Studies (5 runs, Seed 42)**: $5 \text{ runs} \times \approx 30 \text{ seconds/run} \approx 2.5 \text{ minutes}$.
- **Total compute budget**: $\approx 1.5 \text{ hours}$ on the local GPU.

---

## 11. Expected Repository Changes

- [protliggnn_train.py](file:///d:/ProtLigGnn/protliggnn_train.py):
  - Add CLI arguments for `--hidden_dim`, `--dropout`, `--optimizer`, `--scheduler`, `--grad_clip`.
  - Wire optimizer/scheduler creation to command line arguments.
  - Implement LayerNorm disable, residual disable flags.
  - Pass hidden size and dropout to GNN network definition.
