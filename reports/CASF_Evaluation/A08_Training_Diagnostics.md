# A08 Training Diagnostics: CASF-2016

## Training History Analysis (`run_001` / `best_protliggnn.pt`)
- **Total Epochs**: 2
- **Training Samples**: 8
- **Validation Samples**: 1
- **Validation RMSE**: 0.7966
- **Training Loss (MSE)**: 17.12
- **Validation Loss (MSE)**: 0.6434
- **Verdict**: The checkpoint was early-stopped at Epoch 2 because the validation split contained only 1 sample, and the model's random bias prediction of ~3.8 happened to be close to that single sample's target (yielding a tiny validation loss of 0.6434). The model is completely underfitted and has collapsed to a constant predictor due to lack of training data.
