# A09 Root Cause Ranking: CASF-2016

## Ranked Potential Causes
1. **Checkpoint Overwrite Error (Production vs Smoke Checkpoint Mismatch)**
   - **Confidence**: 100%
   - **Evidence**: The file `outputs/best_protliggnn.pt` has `max_samples: 10` and `epochs: 2` in its metadata, confirming it is the smoke-test checkpoint. Evaluating a real production checkpoint like `experiments/run_012/checkpoint.pt` (trained on 1000 samples for 50 epochs) immediately restores normal prediction variance (Std: `0.756` vs `0.074`) and scoring power (Pearson R: `0.418` on a 30-sample subset).
2. **Underfitting/Lack of Converged Training**
   - **Confidence**: 95%
   - **Evidence**: The smoke checkpoint has only been trained for 2 epochs on 8 training samples, so its regression head never learned to correlate structure features with binding affinity.
3. **Distribution Shift**
   - **Confidence**: 5%
   - **Evidence**: PDBbind and CASF-2016 labels are statistically different, but this does not cause prediction collapse since the collapse is present on the training set itself.
