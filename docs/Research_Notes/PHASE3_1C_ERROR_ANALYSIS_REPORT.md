# Phase 3.1C Error Analysis Report
**Experiment ID**: `P3.1C-INTERACTION-REPRESENTATION`  
**Date**: 2026-07-16  

This report details the residual profile and outlier behaviors of the candidate model.

---

## 1. Regression-to-the-Mean Analysis

Linear fit of predicted vs true affinities reveals structural compression:
\[
\hat{y} = 0.3467 \cdot y + 4.0933
\]
- A slope of **0.3467** (less than 1.0) confirms standard regression-to-the-mean, where high-affinity complexes are under-predicted and low-affinity complexes are over-predicted.

---

## 2. Top 10 Best Predictions (Lowest MAE)
- PDB 3qj9: True 7.0000, Pred 7.0042, Error 0.0042
- PDB 2y80: True 8.0458, Pred 8.0501, Error 0.0044
- PDB 3ptg: True 7.6021, Pred 7.6064, Error 0.0044
- PDB 2yig: True 7.1002, Pred 7.1136, Error 0.0450
- PDB 2xo8: True 4.3261, Pred 4.3851, Error 0.0591

---

## 3. Top 10 Worst Predictions (Highest MAE)
- PDB 2xpk: True 11.3010, Pred 6.2491, Error 5.0519
- PDB 3ao4: True 2.0706, Pred 7.0377, Error 4.9671
- PDB 2xiy: True 2.4437, Pred 6.8775, Error 4.4338