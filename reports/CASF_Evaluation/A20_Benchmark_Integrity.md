# A20 Benchmark Integrity: CASF-2016

## Was `outputs/best_protliggnn.pt` the correct checkpoint to benchmark?
**NO**

### Rationale
The checkpoint file `outputs/best_protliggnn.pt` was overwritten by the 2-epoch, 10-sample smoke test checkpoint `run_001` (validation RMSE `0.7966`) during test runs or repository checks. Benchmarking it resulted in a **prediction collapse** because the model weights were underfitted and did not learn structure-affinity correlations.

### Corrective Restoration
The correct production checkpoint that should have been evaluated is `experiments/run_012/checkpoint.pt` (the Seed 42 Phase 6 model). Restoring it to `outputs/best_protliggnn.pt` resolved the prediction collapse and yielded a valid CASF-2016 Pearson correlation of **0.371** and RMSE of **2.111**.