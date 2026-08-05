# Phase 3.1B Experiment Protocol (Revised)
**Experiment ID**: `P3.1B-GEOMETRY-ATTENTION`  
**Baseline Reference**: Certified Optimized Baseline v1.1  
**Status**: Experimental Protocol Locked  

---

## 1. Experimental Variables

### 1.1 Independent Variable
- **Architectural Modification**: Direct addition of distance-based additive bias to cross-attention logits:
  $$S_{ij} \leftarrow S_{ij} + \alpha B_{ij}$$
  where $B_{ij}$ is derived from pairwise ligand-protein Euclidean distances, and $\alpha$ is a learnable scaling parameter represented in log-space:
  $$\alpha = \exp(\log\_alpha)$$
  This guarantees $\alpha > 0$ and provides numerical stability.

### 1.2 Controlled Variables (Strictly Frozen)
- **Graph Preprocessing**: Pre-processed dataset cache (SHA-256: `6fb4385c2...`).
- **Encoders**: Coordinate-free GAT ligand encoder and coordinate-free GCN protein encoder (identical to Baseline v1.1).
- **Regressor Head**: MLP (Layers: `[256, 128, 64, 1]`, identical to Baseline v1.1).
- **Optimizer & Scheduler**: AdamW (lr=0.00019, weight_decay=0.000117, grad_clip=1.0) and CosineAnnealingLR.
- **Training Constraints**: Epochs: 40, Batch Size: 8, Patience: 10.
- **Seeds**: `[42, 123, 777, 2024, 3407]`.
- **Dataset Splits**: 80/10/10 random splits, deterministic per seed (same split IDs).

---

## 2. Statistical Analysis Protocol
- **Paired Two-Sample t-Test**: Primary significance test on test set RMSE across the 5 canonical seeds ($\alpha = 0.05$).
- **Wilcoxon Signed-Rank Test**: Non-parametric test of significance.
- **Cohen's d**: Measure of effect size.
- **Bootstrap 95% Confidence Intervals**: Derived from 1,000 resamples of prediction residuals.

---

## 3. Required Artifacts & Reporting

Every seed run must generate:
- `metrics.json`
- `predictions.csv`
- `training_history.csv`
- `reproducibility_report.json`

The final benchmark aggregator must produce:
- `geometry_attention_metrics.json`: Aggregated metrics.
- `geometry_attention_predictions.csv`: Pooled predictions.
- `geometry_attention_training_history.csv`: Average training curve.
- `interaction_localization_metrics.json`: JSON storing the head-wise correlations, localization scores, and entropy.
- `INTERACTION_LOCALIZATION_REPORT.md`: Comprehensive report with localization analysis.
- `PHASE3_1B_COMPUTATIONAL_PROFILE.md`: Computational profiling report (VRAM, FLOPs, parameter count, checkpoint size, latency).
