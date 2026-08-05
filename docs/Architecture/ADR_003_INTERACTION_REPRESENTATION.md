# ADR 003: Pluggable Interaction Representation Module (IRM)
**Status**: Proposed  
**Date**: 2026-07-15  

---

## 1. Context and Problem Statement
Phase 3.1B demonstrated that injecting distance biases into cross-attention logits yields binding affinity prediction improvements. However, molecular interfaces depend on complex chemistry.

To support subsequent research (auxiliary tasks, contrastive learning, SE(3) equivariance), we must upgrade the interaction block to a unified **Interaction Representation Module (IRM)**. The design must be clean, modular, config-driven, and follow SOLID principles.

---

## 2. Decision
1. **Abstract Interfaces**: Define base classes:
   - `FeatureEncoder`
   - `InteractionFusion`
   - `InteractionCache`
   - `InteractionBiasProjection`
2. **Three-Branch Pipeline**:
   - **Geometry**: Distance and RBF features.
   - **Chemistry**: Atom/residue types, charge, hybridization, aromaticity, donor/acceptor, vdW radius, and electronegativity.
   - **Representation**: Encoded GNN node representations, using a similarity/difference concatenation:
     $$\mathbf{e}^{\text{rep}}_{ij} = [\mathbf{x}_{lig, i} ; \mathbf{x}_{prot, j} ; \mathbf{x}_{lig, i} - \mathbf{x}_{prot, j} ; \mathbf{x}_{lig, i} \odot \mathbf{x}_{prot, j}]$$
3. **Atomic Property Encoder**: Implement `AtomicPropertyEncoder` utilizing RDKit dynamic Periodic Table lookups.
4. **Caching Backend**: Define `InteractionCache` to support pluggable caching (NoCache, MemoryCache, DiskCache).
5. **Decoupled Latent Space**: Maintain the joint `Interaction Embedding` as a standalone latent tensor before projecting to attention logits, allowing auxiliary downstream heads or visualizers to hook into it.

---

## 3. Alternatives Considered

### Alternative A: Monolithic Pairwise MLP
- **Description**: Concatenate all raw properties and embeddings into one vector and feed to a single large MLP.
- **Reason for Rejection**: This violates the Open-Closed Principle and Single Responsibility Principle, and makes it impossible to substitute, disable, or selectively train specific branches.

---

## 4. Consequences
- **Extensibility**: New feature branches or fusion strategies can be introduced simply by adding files and updating the configuration.
- **Explainability**: The joint embedding is fully accessible for down-stream PCA, UMAP, and entropy calculations.
