# Phase 3.1B Verification & Test Plan (Revised)
**Experiment ID**: `P3.1B-GEOMETRY-ATTENTION`  
**Status**: Test Plan Approved  

---

## 1. Test Objectives

The unit tests in [test_geometry_attention.py](file:///d:/ProtLigGnn/tests/test_geometry_attention.py) verify the correctness, numerical stability, and robustness under both CPU and GPU execution modes.

---

## 2. Test Cases

### 2.1 Attention Bias Correctness & Shape
- **Objective**: Verify that `RBFAttentionBias` maps coordinates to correct dimensions.
- **Verification**: Query nodes of length $L$ and key nodes of length $P$ must produce bias tensors of shape `[num_heads, L, P]` (ligand-to-protein) and `[num_heads, P, L]` (protein-to-ligand).

### 2.2 Learnable Log-Scaling Parity
- **Objective**: Assert that the parameter scale is optimized in log-space.
- **Verification**: Checks that the model contains `log_alpha` parameter, that `alpha = torch.exp(log_alpha)` is strictly positive, and that backpropagation gradients flow into `log_alpha`.

### 2.3 Attention Masking Effect
- **Objective**: Verify that passing the attention bias as `attn_mask` changes PyTorch MHA outputs.
- **Verification**: Compare MHA output with zero bias vs. non-zero bias; verify they are numerically different.

### 2.4 Determinism
- **Objective**: Verify that identical seeds yield identical model outputs.
- **Verification**: Run model twice under `torch.no_grad()` on same inputs and assert `torch.allclose(out1, out2, atol=1e-7)`.

### 2.5 Checkpoint Compatibility
- **Objective**: Verify save/load integrity.
- **Verification**: Call `model.state_dict()`, load it into a freshly initialized instance, and verify parameter equality. Specifically verify that the learnable scale `log_alpha` is saved and loaded.

### 2.6 Gradient Flow
- **Objective**: Verify backpropagation gradients flow to trainable parameters.
- **Verification**: Execute a backward pass on random target values. Verify that `model.cross_attention.attention_bias.projector.weight.grad` is not None and contains non-zero elements, and that `model.cross_attention.log_alpha.grad` is not None.

### 2.7 Device Compatibility
- **Objective**: Verify execution on CPU and CUDA.
- **Verification**: Automatically detects CUDA; runs model forward and backward on GPU when available.

---

## 3. Execution Commands
Run the specific test suite:
```bash
pytest tests/test_geometry_attention.py -v
```
Run the full verification suite to check baseline and graph cache integrity:
```bash
pytest tests/ -v
```
