# C04 Regularization Audit: CASF-2016

## Regularization Mechanisms
1. **Dropout**: Currently set to 0.2. Keep at 0.2 for encoders but increase to 0.3 for regressor MLP layers.
2. **Weight Decay**: Currently set to 1e-5. Increase to 1e-4 to penalize large weights and reduce overfitting.
3. **Data Augmentation**: Implement coordinate perturbation (adding normal noise $\sigma = 0.05$Å to coordinates during training) to improve structural robustness.
4. **Exponential Moving Average (EMA)**: Implement EMA on model weights with decay rate `0.999` to stabilize validation predictions.
