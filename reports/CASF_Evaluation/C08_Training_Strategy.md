# C08 Training Strategy: CASF-2016

## Training Strategy Modifications
1. **Stochastic Weight Averaging (SWA)**: Averaging checkpoints from epochs 10–20 is expected to improve validation RMSE by **~0.05**.
2. **Gradient Accumulation**: Accumulating gradients over 4 batches (effective batch size 32) will smooth gradient steps and reduce training variance.
3. **Mixed Precision (AMP)**: Speeds up GNN convolutions on GPU, reducing training epochs latency by ~30%.
