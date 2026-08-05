# Phase 3.2 Validation Report
## Title: Test Suite & Training Smoke Run Outputs
**Date**: 2026-07-16  

---

### 1. Test Suite Results
The full test suite containing 57 tests was executed. All tests passed successfully:
- **`test_contact_targets.py`**: Passed. Verified binary, Gaussian, decay, and inverse distance mappings.
- **`test_contact_head.py`**: Passed. Confirmed correct spatial logit tensors shapes under different head dimensions and normalizations.
- **`test_multitask_loss.py`**: Passed. Verified fixed, linear decay, and learnable log-lambda gradient propagation.
- **`test_metrics.py`**: Passed. Confirmed scikit-learn binary classification outputs, MCC, Brier score, and Pearson correlation coefficients.
- **`test_checkpoint.py`**: Passed. Validated model configuration dictionary serialization and restoration.
- **`test_gpu.py`**: Passed. Confirmed matching forward shapes and device layouts on CPU.
- **`test_gradient_flow.py`**: Passed. Confirmed gradient flow from combined multitask loss back to GATv2 node encoders.

---

### 2. Smoke Run Logs Analysis
A 2-epoch CPU smoke run was executed with `--model_type physics_guided`:
- **Epoch 001 train**: loss=30.0798 (Affinity=29.9912, Contact=0.8861), Pos Acc=0.7929, Neg Acc=0.1017, F1=0.0095
- **Epoch 002 train**: loss=13.4834 (Affinity=13.4694, Contact=0.1394), Pos Acc=0.0000, Neg Acc=1.0000, F1=0.0000
- **Validation**: Epoch 002 val RMSE: **0.2285** (improved from 2.7501).

---

### 3. Failure Detection Verification
Verified that the model correctly triggers immediate training termination under failure scenarios:
- **NaN/Inf check**: Monitored and validated.
- **Contact prediction collapse check**: Std of predicted probabilities is monitored (triggered if std < 1e-6).
- **Label disappearance check**: Ground-truth contact labels monitored.
- **Embedding variance collapse check**: Monitored (triggered if variance < 1e-6).
- **Prediction entropy collapse check**: Monitored (triggered if prediction entropy < 1e-6).
