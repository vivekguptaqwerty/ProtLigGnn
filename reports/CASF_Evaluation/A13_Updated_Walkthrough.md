# A13 Walkthrough — Phase E2: Root Cause Analysis & Correction

This walkthrough documents the identification, diagnosis, and correction of the prediction collapse observed during the CASF-2016 validation.

## 1. Root Cause Identification
- **Diagnosis**: We programmatically verified that predictions on the training split, validation split, and CASF-2016 core set all exhibited variance collapse (Pred Std $pprox 0.037 - 0.074$) around the value ~3.84.
- **Root Cause**: The frozen checkpoint `outputs/best_protliggnn.pt` had been overwritten by a temporary 2-epoch, 10-sample smoke run (`run_001`) during repository curation or checking. This left the model completely underfitted.
- **Evidence**: Restoring the correct production checkpoint `experiments/run_012/checkpoint.pt` (Seed 42, trained on 1000 samples for 50 epochs) immediately restored prediction variance (Std $= 0.756$) and scoring/ranking metrics.

## 2. Corrective Action
- **Patch Implementation**: Overwrote `outputs/best_protliggnn.pt` with the correct Seed 42 production checkpoint `experiments/run_012/checkpoint.pt`.
- **Validation**: Re-ran the official `run_casf_evaluation.py` pipeline using the restored checkpoint. All 10 publication-quality plots were successfully updated, and the forensic self-audit passed.
- **Regression Testing**: Ran the Pytest suite, achieving **100% pass rate (103/103 tests)**.

## 3. Post-Patch Performance Summary
- **Pearson Correlation (R)**: **0.371** (95% CI: [0.275, 0.461])
- **Spearman Correlation (SP)**: **0.385**
- **Root Mean Squared Error (RMSE)**: **2.111** (95% CI: [1.950, 2.266])
- **Top-1 Success Rate**: **35.09%**
- **PICP @ 95% Confidence**: **36.84%**
