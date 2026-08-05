# Interaction Representation Module (IRM) Specification
**Status**: Proposed  
**Version**: 2.0  

This document specifies the refined, multi-branch, and pluggable Interaction Representation Module (IRM).

---

## 1. System Architecture Diagram

```text
       ┌────────────────────────────────────────────────────────┐
       │             Feature Encoders (Pluggable)               │
       │                                                        │
       │  ┌──────────────────┐ ┌────────────────┐ ┌───────────┐ │
       │  │ GeometryEncoder  │ │ChemistryEncoder│ │ RepEncoder│ │
       │  │ (Distance / RBF) │ │(Atom/Res Prop) │ │ (Latents) │ │
       │  └────────┬─────────┘ └───────┬────────┘ └─────┬─────┘ │
       └───────────┼───────────────────┼────────────────┼───────┘
                   ▼                   ▼                ▼
                 [Geo]              [Chem]            [Rep]
                   │                   │                │
                   └───────────────────┼────────────────┘
                                       ▼
                              [InteractionFusion]
                              (Concat, Gated, ...)
                                       │
                                       ▼
                          [Interaction Latent Space]
                           (Hook for downstream use)
                                       │
                                       ▼
                               [InteractionCache]
                              (NoCache, Memory, ...)
                                       │
                                       ▼
                           [InteractionBiasProjection]
                                       │
                                       ▼
                                Attention Bias
```

---

## 2. Abstractions and Pluggable Base Classes

All sub-modules must inherit from these PyTorch base classes:

### 2.1 FeatureEncoder
```python
class FeatureEncoder(nn.Module):
    """Abstract base class for all pairwise feature encoders."""
    def forward(self, ligand_data: dict, protein_data: dict) -> Tensor:
        raise NotImplementedError
```

- **`GeometryFeatureEncoder`**: Exposes spatial relations. Maps coordinates to $L \times P \times D_{\text{geo}}$.
- **`ChemistryFeatureEncoder`**: Models chemical interactions. Exposes $L \times P \times D_{\text{chem}}$.
- **`RepresentationFeatureEncoder`**: Processes GNN node embeddings. Computes:
  $$\mathbf{e}^{\text{rep}}_{ij} = [\mathbf{x}_{lig, i} ; \mathbf{x}_{prot, j} ; \mathbf{x}_{lig, i} - \mathbf{x}_{prot, j} ; \mathbf{x}_{lig, i} \odot \mathbf{x}_{prot, j}]$$

### 2.2 InteractionFusion
```python
class InteractionFusion(nn.Module):
    """Abstract base class for fusing branch embeddings."""
    def forward(self, geo: Tensor, chem: Tensor, rep: Tensor) -> Tensor:
        raise NotImplementedError
```
- Concretely implemented as:
  - `ConcatFusion`: Direct concatenation and linear projection.
  - `GatedFusion`: Gated Highway / GRU-like selection.
  - `FiLMFusion`: Feature-wise Linear Modulation.
  - `TransformerFusion`: Cross-attention over branch tokens.

### 2.3 InteractionCache
```python
class InteractionCache(nn.Module):
    """Abstract base class for interaction representation caching."""
    def get(self, key: str) -> Optional[Tensor]:
        raise NotImplementedError
    def set(self, key: str, value: Tensor) -> None:
        raise NotImplementedError
```
- Concretely implemented as:
  - `NoCache`: Returns `None` immediately.
  - `MemoryCache`: In-memory dictionary.
  - `DiskCache`: File-based caching.
