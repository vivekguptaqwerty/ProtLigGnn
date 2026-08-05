# Phase 3.1B Implementation Report: Distance-Biased Cross-Attention
**Experiment ID**: `P3.1B-GEOMETRY-ATTENTION`  
**Status**: Implementation Complete  
**Date**: 2026-07-15  

---

## 1. Goal and Mathematical Formulation
This experiment isolates a single independent variable compared to the certified `Optimized Baseline v1.1`: injecting physical, Euclidean spatial distance bias directly into the cross-attention interaction block.

### 1.1 Logarithmic Scale Parameter ($\alpha$)
Attention weights are biased additively and scaled via a log-space parameter:
$$\text{AttentionWeights} = \text{Softmax}\left(S_{ij} + \alpha B_{h, i, j}\right)$$
$$\alpha = \exp(\log\_alpha)$$
$$\log\_alpha = nn.Parameter(torch.tensor(math.log(0.1)))$$

This guarantees $\alpha > 0$ at all times, avoids sign inversion, and ensures smoother optimization gradients.

---

## 2. Codebase layout

### 2.1 Files Added
1. [interfaces.py](file:///d:/ProtLigGnn/models/geometry_attention/interfaces.py): Defines the abstract `AttentionBias` interface.
2. [config.py](file:///d:/ProtLigGnn/models/geometry_attention/config.py): Implements the `AttentionBiasConfig` dataclass supporting full config-driven serialization and checkpointing.
3. [geometry_attention_utils.py](file:///d:/ProtLigGnn/models/geometry_attention/geometry_attention_utils.py): Vectorized pairwise Euclidean distance calculation helper functions.
4. [attention_bias.py](file:///d:/ProtLigGnn/models/geometry_attention/attention_bias.py): Implements `RBFAttentionBias` (Gaussian RBF projected to attention heads) and a placeholder `LearnedAttentionBias` illustrating pluggability.
5. [geometry_attention_model.py](file:///d:/ProtLigGnn/models/geometry_attention/geometry_attention_model.py): Implements `DistanceBiasedCrossAttention` layer and `ProtLigGNNGeometryAttention` wrapper model.
6. [test_geometry_attention.py](file:///d:/ProtLigGnn/tests/test_geometry_attention.py): Shape and forward/backward gradient tests.
7. [test_attention_bias.py](file:///d:/ProtLigGnn/tests/test_attention_bias.py): Test shapes and RBF values.
8. [test_localization_metrics.py](file:///d:/ProtLigGnn/tests/test_localization_metrics.py): Test entropy and spatial correlation calculations.
9. [test_geometry_attention_checkpoint.py](file:///d:/ProtLigGnn/tests/test_geometry_attention_checkpoint.py): Checkpoint saving, loading, and parameter preservation tests.

### 2.2 Files Modified
- [protliggnn_train.py](file:///d:/ProtLigGnn/protliggnn_train.py): Integrates `--model_type geometry_attention` and serializes `attention_bias_config` inside checkpoint payload.
- [benchmark.py](file:///d:/ProtLigGnn/benchmark.py): Integrates model choices in argparser and forwards attention parameters.

---

## 3. Verification & Validation Summary
- **Unit Tests**: All 30 tests in the repository pass. The new `test_geometry_attention` suite completes successfully, verifying gradient flow to `log_alpha` and the linear projector weights.
- **CPU Smoke Run**: Completed a 2-epoch run, verifying that checkpoints are saved and loaded correctly under training configurations.
