# A10 Corrective Action Plan: CASF-2016

## Action Plan
- **Confirmed Root Cause**: Checkpoint `outputs/best_protliggnn.pt` was overwritten by a temporary smoke test during verification / repository curation.
- **Corrective Patch**: Copy the correct Phase 6 frozen production checkpoint `experiments/run_012/checkpoint.pt` (trained on 1000 samples for 50 epochs with Seed 42) back to `outputs/best_protliggnn.pt`.
- **Files Requiring Modification**: None (file system copy command).
- **Engineering Effort**: < 1 minute.
- **Expected Outcome**:
  - Pearson R: ~0.40 - 0.50
  - Prediction Std: ~0.70 - 0.85
  - RMSE: ~1.8 - 2.1
