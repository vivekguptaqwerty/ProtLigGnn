# Phase 3.1A Official Benchmark Report
**Experiment ID**: P3.1A-GEOMETRY-LIGAND  
**Model Under Test**: ProtLigGNNGeometry (Invariant Ligand Geometry Feature Injection)  
**Baseline Model**: Certified Optimized Baseline v1.1  
**Execution Date**: 2026-07-12  
**Status**: Benchmark Completed Successfully  

---

## 1. Executive Summary
This report presents the official benchmark results for Phase 3.1A. The experiment investigated whether injecting explicit geometric edge representations (Euclidean distances mapped via a 32-center Gaussian Radial Basis Function and projected to 32 dimensions) into the GATConv layers of the ligand encoder improves affinity prediction over the certified Baseline v1.1.

The benchmark was executed across all **5 canonical seeds** using the frozen dataset ($N=1,000$), split protocol, and training hyperparameters. 

---

## 2. Benchmark Configuration
- **Model Type**: `geometry`
- **Ligand Encoder**: GATConv upgraded with 32D explicit geometric edge attributes.
- **Protein Encoder**: GCNConv, coordinate-free (identical to Baseline v1.1).
- **Cross-Attention & Readout**: Identical to Baseline v1.1.
- **Optimizer**: AdamW (lr=0.00019, weight_decay=0.000117, grad_clip=1.0)
- **Scheduler**: CosineAnnealingLR (epochs=40, patience=10)
- **Seeds**: `[42, 123, 777, 2024, 3407]`
- **Dataset / Cache**: PDBbind v2020 Local Subset, validated SHA-256: `6fb4385c2...`
- **Splits**: 80% train (800), 10% val (100), 10% test (100)

---

## 3. Per-Seed Execution Metrics

| Seed | Run Directory | Best Epoch | Test RMSE | Test MAE | Test PCC | Test Spearman | Test $R^2$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | `experiments/run_047` | 13 | 1.4616 | 1.1502 | 0.5641 | 0.5926 | 0.2977 |
| **123** | `experiments/run_048` | 13 | 1.5727 | 1.2162 | 0.5722 | 0.5213 | 0.2865 |
| **777** | `experiments/run_049` | 29 | 1.5389 | 1.2429 | 0.5361 | 0.5250 | 0.2370 |
| **2024** | `experiments/run_050` | 13 | 1.6671 | 1.3021 | 0.5280 | 0.5491 | 0.2740 |
| **3407** | `experiments/run_051` | 29 | 1.4810 | 1.1582 | 0.5348 | 0.5305 | 0.2223 |

---

## 4. Aggregated Benchmark Statistics
We aggregate the results across the 5 runs and compute standard error and 95% confidence intervals (CI):

| Metric | Mean $\pm$ Std | 95% CI | Median | Min | Max |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RMSE** | $1.5442 \pm 0.0818$ | $[1.4726, 1.6159]$ | $1.5389$ | $1.4616$ | $1.6671$ |
| **MAE** | $1.2139 \pm 0.0628$ | $[1.1589, 1.2690]$ | $1.2162$ | $1.1502$ | $1.3021$ |
| **PCC** | $0.5470 \pm 0.0197$ | $[0.5297, 0.5643]$ | $0.5361$ | $0.5280$ | $0.5722$ |
| **Spearman** | $0.5437 \pm 0.0293$ | $[0.5180, 0.5694]$ | $0.5305$ | $0.5213$ | $0.5926$ |
| **$R^2$** | $0.2635 \pm 0.0324$ | $[0.2351, 0.2919]$ | $0.2740$ | $0.2223$ | $0.2977$ |

---

## 5. Artifact Validation & Verification
All runs successfully completed, saving the checkpoints, predictions, training history, and environment locks. Required outputs were verified before registering:
- Checkpoints saved: Yes.
- Predictions CSV: [geometry_predictions.csv](file:///d:/ProtLigGnn/experiments/benchmark_2026-07-12T203928_0000/geometry_predictions.csv) contains 500 total test predictions.
- Metrics JSON: [geometry_metrics.json](file:///d:/ProtLigGnn/experiments/benchmark_2026-07-12T203928_0000/geometry_metrics.json).
- Training History: [geometry_training_history.csv](file:///d:/ProtLigGnn/experiments/benchmark_2026-07-12T203928_0000/geometry_training_history.csv).
