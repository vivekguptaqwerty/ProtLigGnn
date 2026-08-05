# Module 3 — Hyperparameter Optimization (HPO) Report

**Study Name**: `protliggnn_hpo_v2_1`  
**Search Method**: Bayesian Optimization (TPE Sampler via Optuna)  
**Total Completed Trials**: 20  
**Search Objective**: Minimize validation RMSE (on seed 42)  
**Analysis Date**: 2026-07-12  

---

## 1. Executive Summary

This report documents the staged hyperparameter optimization (HPO) study designed to maximize the representation capacity of the existing ProtLigGNN model. 

By running a Bayesian search over 20 completed trials on seed 42, we identified key trends:
- Schedulers (specifically `cosine` and `plateau`) are critical to preventing premature stagnation of training.
- Optimizers with weight decay (`adamw` and `radam`) outperform the standard `adam` baseline.
- Models with smaller hidden sizes (`hidden_dim: 128` or `256`) yield equal or better validation results than larger networks (`512`), while having significantly faster training speed and smaller memory footprints.

---

## 2. Optimization History & Parameter Importance

We visualized the study using matplotlib plots:

- **Optimization History**: The search reached a low validation RMSE of **1.3286** at Trial 0, and consistently explored values under 1.40 throughout the study.

![HPO History](file:///C:/Users/Harshhhh/.gemini/antigravity/brain/03c21182-9e56-4764-a6fd-83660d6d0fc1/optuna_history.png)

- **Hyperparameter Importance**:

![Parameter Importance](file:///C:/Users/Harshhhh/.gemini/antigravity/brain/03c21182-9e56-4764-a6fd-83660d6d0fc1/optuna_importance.png)

---

## 3. Top 3 Finalists Selection

From the 20 completed trials, the top 3 finalists selected for multi-seed benchmark validation are:

### Finalist 1 (Trial 0) — *The Champion*
- **Validation RMSE**: **1.3286**
- **Parameters**:
  - `hidden_dim`: 128
  - `dropout`: 0.3181
  - `lr`: $1.90 \times 10^{-4}$
  - `weight_decay`: $1.17 \times 10^{-4}$
  - `optimizer`: `adamw`
  - `scheduler`: `cosine`
  - `grad_clip`: 1.0

### Finalist 2 (Trial 7) — *The Challenger*
- **Validation RMSE**: **1.3613**
- **Parameters**:
  - `hidden_dim`: 256
  - `dropout`: 0.2028
  - `lr`: $2.22 \times 10^{-4}$
  - `weight_decay`: $8.25 \times 10^{-5}$
  - `optimizer`: `radam`
  - `scheduler`: `plateau`
  - `grad_clip`: 10.0

### Finalist 3 (Trial 3) — *The Runner-Up*
- **Validation RMSE**: **1.3711**
- **Parameters**:
  - `hidden_dim`: 256
  - `dropout`: 0.1766
  - `lr`: $3.34 \times 10^{-4}$
  - `weight_decay`: $4.91 \times 10^{-5}$
  - `optimizer`: `adamw`
  - `scheduler`: `plateau`
  - `grad_clip`: 10.0
