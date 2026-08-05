# Phase 3.1C Implementation Plan: Interaction Representation Module
**Experiment ID**: `P3.1C-EDGE-AWARE-INTERACTION`  
**Status**: Ready for Approval  

This implementation plan details the architectural designs, class structures, configuration schemas, testing strategies, and audit procedures for Phase 3.1C.

---

## 1. Refined Architectural Design

The **Interaction Representation Module (IRM)** replaces simple distance-only attention bias with a multi-branch representation learning architecture. The system is designed to be fully compliant with SOLID and Clean Architecture principles.

```text
Interaction Representation
├── Feature Encoders
│      ├── Geometry (RBF distances)
│      ├── Chemistry (raw atom properties via AtomicPropertyEncoder)
│      └── Representation (encoded GNN node embeds)
│
├── Feature Fusion (Concat, Gated, Add, etc.)
│
├── Interaction Latent Space
│
├── Interaction Cache (NoCache, Memory, Disk)
│
├── Bias Projection (Linear projection)
│
└── Cross Attention (biasing logit additions)
```

---

## 2. Pluggable Interfaces Spec
```python
class FeatureEncoder(nn.Module):
    def forward(self, ligand_data: dict, protein_data: dict) -> Tensor:
        raise NotImplementedError

class InteractionFusion(nn.Module):
    def forward(self, geo: Tensor, chem: Tensor, rep: Tensor) -> Tensor:
        raise NotImplementedError

class InteractionCache(nn.Module):
    def get(self, key: str) -> Optional[Tensor]:
        raise NotImplementedError
    def set(self, key: str, value: Tensor) -> None:
        raise NotImplementedError
```

---

## 3. Configuration System & Checkpoint Compatibility
The `InteractionConfig` class governs the IRM:
```python
@dataclass
class InteractionConfig:
    schema_version: str = "1.0"
    interaction_dim: int = 32
    geometry_dim: int = 32
    chemistry_dim: int = 32
    representation_dim: int = 32
    geometry_encoder: str = "rbf"
    chemistry_encoder: str = "linear_prop"
    representation_encoder: str = "concat_project"
    fusion_strategy: str = "concat"
    bias_projection: str = "linear"
    dropout: float = 0.1
    activation: str = "relu"
    normalization: str = "layer"
    cache_backend: str = "memory"
    num_rbf: int = 32
    cutoff_distance: float = 12.0
    learnable_scale: bool = True
    initial_alpha: float = 0.1
    normalize_bias: bool = True
```
Checkpoints will serialize the `InteractionConfig` fields. Legacy models will resolve fallback configurations cleanly to preserve backward compatibility.

---

## 4. Robust Stopping Rules
During training, the system will monitor and trigger early termination if any of the following failure conditions is detected:
1. **NaN Loss**: Loss values become `NaN` or `Inf`.
2. **Gradient Explosion**: L2 norm of the gradients exceeds $1000.0$.
3. **Entropy Collapse**: Mean attention entropy across heads falls below $0.001$, indicating collapsing focus.
4. **Embedding Collapse**: Standard deviation of the interaction embeddings falls below $1e-5$ (model degeneration).
5. **Bias Variance Collapse**: Variance of the generated attention biases falls below $1e-6$ (dead bias predictions).
6. **Dead Attention Heads**: Variance of attention weights in a head is zero.

---

## 5. Verification Plan

### Automated Test Matrix
- `test_interaction_bias.py`: Test RBF, Chemistry, and Representation branches shapes.
- `test_pairwise_features.py`: Test `AtomicPropertyEncoder` lookup correctness.
- `test_geometry_interaction.py`: Verify forward flow, gradient propagation, and GPU compatibility.
- `test_interaction_checkpoint.py`: Check serialization, config parsing, and compatibility.
