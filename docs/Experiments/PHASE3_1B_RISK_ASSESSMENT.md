# Phase 3.1B Risk Assessment
**Experiment ID**: `P3.1B-GEOMETRY-ATTENTION`  
**Status**: Risk Assessment Audited  

This report identifies potential technical and performance risks of the distance-biased cross-attention design and provides clear mitigations.

---

## 1. Technical Risk Matrix

### Risk 1: Gradient Instability & Exploding Logits
- **Description**: Additive attention biases that are too large in magnitude can cause the attention weights to saturate (all weights close to $0.0$ or $1.0$). This results in vanishing gradients during backward passes.
- **Likelihood**: Medium
- **Impact**: High
- **Mitigation**: 
  1. We initialize the scaling parameter $\alpha$ to a small constant (e.g., $0.1$), ensuring distance biases start as subtle corrections.
  2. We provide an option `normalize_bias` in `AttentionBiasConfig` that applies LayerNorm to the projected bias matrices, keeping them bounded.

### Risk 2: Training Speed & Computation Bottleneck
- **Description**: Computing pairwise coordinates distance matrices $D \in \mathbb{R}^{L \times P}$ and expanding them via $K$ basis functions inside nested batch loops can increase GPU overhead.
- **Likelihood**: Low
- **Impact**: Medium
- **Mitigation**:
  - The local dataset complexes are small: ligands have $\le 100$ atoms, protein pockets have $\le 50$ residues.
  - Distance computation is vectorized using PyTorch's native operations, which runs efficiently on both CPU and GPU.

### Risk 3: Spatial Generalization Overfitting
- **Description**: The learnable projection layer mapping RBF outputs to the attention heads adds weights that could overfit the small local training subset.
- **Likelihood**: Medium
- **Impact**: Medium
- **Mitigation**:
  - The parameter count is extremely small: $\text{Linear}(32, 4)$ adds only 132 weights.
  - Dropout inside MultiheadAttention is maintained at $0.159$ (matching Baseline v1.1's MHA dropout), acting as a strict regularizer.
