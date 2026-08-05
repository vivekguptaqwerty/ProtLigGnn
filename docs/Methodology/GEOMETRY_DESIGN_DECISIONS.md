# Geometry Design Decisions: Phase 3.1A Architecture
**Status**: Immutable Decision Record  
**Milestone**: Phase 3.1A  

This document records the design decisions and scientific trade-offs made during the introduction of explicit geometric features in **ProtLigGNN Phase 3.1A**.

---

## 1. Rationale for Radial Basis Functions (RBF)
We map continuous distances $d_{ij}$ to high-dimensional representations using Gaussian Radial Basis Functions (RBF) because:
- **Smoothness & Differentiability**: Unlike discrete distance bins (which introduce step discontinuities that break gradient propagation), RBFs represent continuous distances smoothly and are fully differentiable.
- **Local Expressive Capability**: Gaussians localize distance information. If an edge distance is $1.5 \text{ Å}$, the RBF activates centers close to $1.5 \text{ Å}$ while leaving distant centers at zero, providing structural granularity.
- **Standard Practice**: RBF expansions are the standard method in geometric deep learning (e.g., SchNet, PhysNet, DimeNet) for mapping physical bond lengths.

---

## 2. Choosing 32 Basis Functions
- **Resolution**: 32 basis functions spread over $[0.0, 12.0] \text{ Å}$ yield a spacing of $\approx 0.38 \text{ Å}$ between adjacent centers. This provides sufficient spatial resolution to distinguish bond lengths and close spatial contacts.
- **Capacity Trade-off**: Increasing basis count (e.g., to 64 or 128) increases representation capacity but also increases the parameter count of the projection layers and GAT attention keys. 32 functions provide the optimal balance of resolution and computational speed.

---

## 3. Isolating Geometry to the Ligand Encoder Only
- **Controlled Variable**: The protein pocket encoder remains 100% identical to the GCN encoder in Baseline v1.1.
- **Native GAT Support**: The ligand encoder is built using GATConv, which natively supports multidimensional `edge_dim`. 
- **Scientific Cleanliness**: If we modified both encoders simultaneously (e.g., converting GCN to a geometry-aware layer or learning scalar weights), any performance delta would be due to a combination of ligand geometry and protein geometry changes. By isolating geometry to the ligand encoder, we answers a single, precise scientific question: *Does explicit ligand-only geometry improve representations?* Protein pocket geometry will be investigated independently in Phase 3.1C.

---

## 4. Preserving the Certified Baseline GAT/GCN Structure
- **No Replacement**: We preserve the baseline's convolutional layers (GATConv/GCNConv) and hidden states. No advanced architectures (like EGNN, Equiformer, or Transformers) are introduced.
- **Direct Comparison**: Preserving the baseline layers guarantees that any change in metrics is caused solely by the introduction of explicit distance attributes.

---

## 5. Omission of Angles, Torsions, and Equivariance
- **Angles & Torsions**: Computing 3-body angles or 4-body dihedral/torsion angles requires significant computational overhead ($O(N^3)$ and $O(N^4)$ complexity). It also introduces multiple structural variables. 
- **Equivariance**: Equivariant networks (such as EGNN, PaiNN, or SE(3)-Transformers) update 3D coordinate tensors directly. This represents a complete paradigm shift from coordinate-free graphs. In Phase 3.1A, we focus strictly on **invariant edge attributes** (Euclidean distances are invariant to 3D rotation and translation). Equivariant coordinates will be studied in later milestones.

---

## 6. Preserving Benchmark Compatibility
- **Frozen Pipeline**: The benchmark protocol, seeds, split lists, dataset extraction, optimizer, and scheduler remain unchanged. This preserves full comparability against the certified Baseline v1.1 metrics.
