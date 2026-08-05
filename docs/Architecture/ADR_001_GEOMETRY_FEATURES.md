# Architecture Decision Record (ADR) 001: Explicit Invariant Ligand Geometry
**Date**: 2026-07-12  
**Status**: Approved  
**Author**: Lead Geometric Deep Learning Scientist  

---

## 1. Context and Problem Statement
ProtLigGNN Baseline v1.1 processes ligand and protein complexes as topological graphs without explicit geometric edge features. While the dataset contains 3D spatial coordinates, these are only used to filter pocket residues within a 12Å radius. Inside GNN convolutions, representation learning remains coordinate-free, leading to weak correlation between learned attention weights and physical spatial distances. We need to introduce explicit geometric representations without breaking benchmark equivalence or baseline reproducibility.

---

## 2. Proposed Decision
We introduce explicit, rotation- and translation-invariant 3D Euclidean distances into the **ligand encoder** of ProtLigGNN using:
1. Dynamic distance computation from conformer coordinates on the fly.
2. A 32-dimensional Gaussian Radial Basis Function (RBF) expansion spanning $0.0 \text{ Å} \to 12.0 \text{ Å}$.
3. A linear projection mapping RBF outputs to 32D edge attributes.
4. PyTorch Geometric GATConv layers in the ligand encoder configured to ingest these edge attributes.

The protein encoder (GCNConv) remains untouched. The training script will select the model type via a `--model_type` CLI option.

---

## 3. Alternatives Considered

### Alternative A: Coordinate-Aware Equivariant GNN (e.g., EGNN)
- **Description**: Replace GATConv and GCNConv with equivariant convolutions that update node coordinates $\vec{x}_i \leftarrow \vec{x}_i + \sum \phi_x(h_i, h_j, d_{ij}^2) (\vec{x}_i - \vec{x}_j)$.
- **Reason for Rejection**: This requires replacing both encoders, which breaks the strict requirement to change only one scientific variable. Equivariance represents a major architectural paradigm shift that will be evaluated in a later phase.

### Alternative B: Learned Scalar Edge Weights for GCN Conv
- **Description**: Project ligand and protein RBF features down to scalar weights and pass them as `edge_weight` to both GATConv and GCNConv.
- **Reason for Rejection**: Modifying both encoders at the same time prevents scientific isolation. PyG's GCNConv does not natively support multi-dimensional edge features, and compressing them to a 1D scalar loses valuable representation capacity.

---

## 4. Consequences
- **Codebase Separation**: Baseline code remains 100% frozen. All new geometric logic is isolated inside `models/geometry/`.
- **Parameter Overhead**: Adds a modest 1,120 parameters to the model (for the distance projection layer), representing a negligible 0.3% increase in parameters.
- **Runtime Performance**: Distance calculation on the fly adds small computational overhead ($< 1 \text{ ms}$ per batch), preserving high virtual screening throughput.
- **Reproducibility**: The baseline model remains perfectly reproducible by running `--model_type baseline`.

---

## 5. Future Roadmap
- **Phase 3.1B**: Introduce distance-biased cross-attention (biasing multi-head query-key products using inter-graph RBF distances).
- **Phase 3.1C**: Extend explicit geometry to the protein pocket GCN encoder using invariant distance weights.
- **Phase 3.2**: Investigate coordinate equivariance using E(3)-invariant messages and equivariant updates.
