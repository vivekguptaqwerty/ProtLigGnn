# B02 Learning Curve Analysis: CASF-2016

## Training Dynamics Recovery
- **Best Epoch**: 13
- **Validation RMSE at Best Epoch**: 1.4049
- **Training RMSE at Best Epoch**: 1.7228
- **Generalization Gap (RMSE)**: -0.3179

## Overfitting & Convergence Summary
- The validation loss hits a minimum at **epoch 13**, after which it diverges significantly (e.g. rising to `3.60` and `4.48`), while the training loss continues to drop consistently from `4.95` to `2.19`.
- This represents a classic signature of **severe overfitting**. The model memorized the small training set of 800 samples within 13 epochs, limiting its ability to generalize to the external CASF-2016 benchmark.
