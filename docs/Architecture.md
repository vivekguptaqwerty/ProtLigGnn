# System Architecture — LigProtGNN-X v4.0

This document consolidates system design parameters for `LigProtGNN-X v4.0`.

## 1. Core Model
- **Protein Encoder**: Frozen ESM-2 protein foundation model.
- **Ligand Encoder**: Frozen ChemBERTa ligand foundation model.
- **Bidirectional Cross Attention**: Connects protein and ligand latents.
- **Geometry Attention**: Pluggable distance-biased radial basis functions (RBF) for spatial contact mapping.
- **Affinity Head**: Multi-layer MLP mapping joint representation coordinates to regressed binding affinity (pK).

## 2. Integrated Modules
- **Reliability Layer**: Variance and temperature scaling; Evidential Normal-Inverse-Gamma GNN regressor head.
- **Explainability Layer**: Integrated Gradients node embeddings attributions; Hook-based Cross-Attention Rollout.
- **Robustness Suite**: Composite Robustness Index (CRI) sweeps covering coordinate noise and OOD shifts.