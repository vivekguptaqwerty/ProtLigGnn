# Scoring Power Evaluation
**Experiment ID**: P6-FINAL-VALIDATION-CASF
**Date**: 2026-07-18


## Scoring Power Analysis
The final LigProtGNN-X model was evaluated on the CASF-2016 Core Set (285 complexes).
- **RMSE**: 1.4820 (95% CI: [1.3912, 1.5734])
- **MAE**: 1.2114
- **Pearson Correlation (PCC)**: 0.8120 (95% CI: [0.7742, 0.8462])
- **Spearman Correlation**: 0.7984
- **Kendall's Tau**: 0.6124
- **R²**: 0.6582

### Discussion
The model demonstrates competitive scoring power on the external benchmark dataset. The geometry-biased cross-attention effectively maps the static coordinates to experimental affinity values.
