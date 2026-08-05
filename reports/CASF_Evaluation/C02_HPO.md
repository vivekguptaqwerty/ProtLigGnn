# C02 Hyperparameter Optimization: CASF-2016

## Recommended Hyperparameter Search Bounds
- **Learning Rate**: [1e-4, 5e-3] (Log-uniform)
- **Weight Decay**: [1e-6, 1e-3] (Log-uniform)
- **Dropout**: [0.1, 0.5] (Uniform)
- **Batch Size**: [16, 64] (Discrete categorical)
- **LR Scheduler**: Cosine Annealing (with Warmup epochs [1, 5])
- **Gradient Clipping**: Max norm [1.0, 5.0]

## Search Strategy
Recommend using **Optuna** with Tree-structured Parzen Estimators (TPE) algorithm. Optuna allows pruning unpromising trials early using MedianPruner, yielding a high-throughput search over 100 trials within 24 hours on the RTX 3050 GPU.
