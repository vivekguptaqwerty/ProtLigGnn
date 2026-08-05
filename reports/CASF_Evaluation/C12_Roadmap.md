# C12 Roadmap: CASF-2016 Validation

## Prioritized Recovery Roadmap

### Phase 1: Quick Wins (1-2 days)
- Copy all Refined PDBbind complexes to training cache, expanding train set to N=4,300.
- Implement Cosine Annealing learning rate scheduler with warmup in `protliggnn_train.py`.
- **Expected Metrics**: Pearson R $\rightarrow$ **0.460**, RMSE $\rightarrow$ **1.850**

### Phase 2: Medium Improvements (1 week)
- Run Optuna HPO search (100 trials) to find optimal learning rates, weight decays, and dropout bounds.
- Tighten pocket extraction cutoff to 4.5Å.
- **Expected Metrics**: Pearson R $\rightarrow$ **0.500**, RMSE $\rightarrow$ **1.780**

### Phase 3: Major Improvements (2-4 weeks)
- Implement SWA (Stochastic Weight Averaging) on best epochs.
- Add physical features (Hydrogen Bonds and SASA hydrophobic surface calculations).
- **Expected Metrics**: Pearson R $\rightarrow$ **0.540**, RMSE $\rightarrow$ **1.690**
