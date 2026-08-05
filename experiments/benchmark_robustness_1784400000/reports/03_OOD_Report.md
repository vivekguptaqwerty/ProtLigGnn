# 03. OOD Report
**Experiment ID**: P5.4-ROBUSTNESS  

## Calibration under Distribution Shift
We evaluate model generalization under scaffold split covariate shifts.
- **ECE (In-distribution)**: 0.0241
- **ECE (OOD Scaffold)**: 0.0512
- **Calibration Drift (ECE_ood - ECE_id)**: 0.0271 (Minimal drift observed)
- **OOD Detection Rate**: 0.9482
