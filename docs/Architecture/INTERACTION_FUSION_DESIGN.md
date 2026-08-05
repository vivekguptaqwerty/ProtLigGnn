# Interaction Fusion Design
**Status**: Proposed  
**Version**: 1.0  

This document details the strategies for fusing the multi-branch embeddings (geometry, chemistry, and representation) inside the IRM.

---

## 1. Fusion Strategies

### 1.1 Concatenation Fusion (Selected for Phase 3.1C)
The embeddings are concatenated along the last dimension and projected to the target joint dimension:
$$\mathbf{z}_{ij} = \text{ReLU}\left(\mathbf{W}_f \cdot [\mathbf{e}^{\text{geo}}_{ij} ; \mathbf{e}^{\text{chem}}_{ij} ; \mathbf{e}^{\text{rep}}_{ij}] + \mathbf{b}_f\right)$$
- **Benefits**: Simple, mathematically stable, preserves raw feature magnitudes, and facilitates easy gradient flow.

### 1.2 Multiplicative / Bilinear Fusion (Future Phase)
Geometry acts as a physical gating matrix over chemical and representational features:
$$\mathbf{z}_{ij} = \mathbf{e}^{\text{geo}}_{ij} \odot \left(\mathbf{W}_c \mathbf{e}^{\text{chem}}_{ij}\right) \odot \left(\mathbf{W}_r \mathbf{e}^{\text{rep}}_{ij}\right)$$
- **Benefits**: Physically realistic; chemical interactions are gated to zero when the distance is beyond the cutoff limit.

---

## 2. Pluggable Factory Mapping

The configuration system dynamically loads the fusion implementation via the `fusion_strategy` parameter:
- `fusion_strategy: "concat"` -> Instantiates `ConcatenationFusion`.
- `fusion_strategy: "multiplicative"` -> Instantiates `MultiplicativeFusion`.
