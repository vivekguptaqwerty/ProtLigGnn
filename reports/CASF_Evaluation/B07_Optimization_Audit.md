# B07 Optimization Audit: CASF-2016

## Hyperparameters in checkpoint metadata
- **Optimizer**: Adam
- **Learning Rate**: 1e-3
- **Weight Decay**: 1e-5
- **Scheduler**: None
- **Batch Size**: 8
- **Dropout**: 0.2
- **Verdict**: The training run did not use a learning rate scheduler, warmup, or gradient clipping. Adam with a constant LR of 1e-3 on batch size 8 leads to unstable optimization steps on small datasets, accelerating overfitting.
