# Geometry Feature Specification: Coordinate & Edge Distance Encoding
**Specification Version**: v1.0  
**Status**: Canonical Reference Document  
**Milestone**: Phase 3.1A  

This document specifies the coordinate representation, edge distance extraction, numerical precision standards, and mapping algorithms for explicit geometric features in the **ProtLigGNN** architecture.

---

## 1. Coordinate Source & Extraction
- **Ligand Coordinates**: Derived dynamically from the RDKit `Conformer` of the SDF/MOL2 ligand molecule file. Only 3D coordinates are allowed. If the ligand lacks a 3D conformer, it is skipped.
- **Protein Coordinates**: Pocket residues coordinates are extracted from PDB files. The coordinate is centered on the alpha-carbon ($C_\alpha$) of each pocket residue. If $C_\alpha$ is missing, the mean of the residue's non-hydrogen atoms is used.
- **Unit Measurement**: Ångströms (Å) for all spatial coordinates and distances.
- **Numerical Precision**: Single-precision floating point (`float32`) is enforced throughout the geometric pipeline to prevent numerical drift across CPU/GPU boundaries.

---

## 2. Edge Distance Calculation
For every directed edge $(i, j)$ in the graph:
1. Extract coordinates $\vec{p}_i, \vec{p}_j \in \mathbb{R}^3$.
2. Compute Euclidean distance $d_{ij} = \|\vec{p}_i - \vec{p}_j\|_2$.
3. Distances must be strictly positive: $d_{ij} \ge 0$. Covalent bond distances range from $\approx 1.0 \text{ Å}$ to $\approx 2.5 \text{ Å}$.

---

## 3. Gaussian Radial Basis Function (RBF) Mapping
Continuous distances $d_{ij}$ are projected into a discrete bank of 32 Gaussian kernels:
$$e_k(d_{ij}) = \exp\left(-\frac{(d_{ij} - \mu_k)^2}{2\sigma^2}\right)$$
where:
- **Basis Count ($K$)**: 32
- **Distance Range**: $0.0 \text{ Å} \to 12.0 \text{ Å}$
- **Centers ($\mu_k$)**: Linearly spaced values:
  $$\mu_k = \frac{12.0 \cdot k}{K - 1}, \quad k \in \{0, \dots, 31\}$$
- **Bandwidth ($\gamma$)**: Defined using the spacing between adjacent centers:
  $$\sigma = \mu_1 - \mu_0 = \frac{12.0}{31} \approx 0.3871, \quad \gamma = \frac{1}{2\sigma^2} \approx 3.337$$

---

## 4. Self-Loop and Boundary Handling
- **Self-Loops**: Self-loops are handled separately. When computing GATConv or GCNConv convolutions, self-loops are handled natively by the PyTorch Geometric layer logic (either by adding self-loop features or using a separate weight matrix).
- **Missing Coordinates**: Checked during dataloading. Any complex missing conformers or coordinates is rejected during PDBbind parsing and recorded in skip logs.
- **Maximum Cutoff**: The RBF range is capped at $12.0 \text{ Å}$. Distances beyond this range fall off smoothly to near-zero value in RBF space due to the exponential decay of the Gaussian kernels.

---

## 5. Edge Attribute Projections & Dimensions
- **Input Dimension**: 32 (output of the Gaussian RBF expansion).
- **Projection Layer**: A single linear layer mapping 32 dimensions to 32 dimensions:
  $$\text{edge\_attr}_{ij} = \mathbf{W}_{lig} \cdot e(d_{ij}) + \mathbf{b}_{lig}$$
- **Output Dimension**: 32-dimensional continuous vector representing ligand edge features.
- **Target Encoder**: Feed directly into `GeometryAwareLigandEncoder` GATConv layers:
  $$\text{GATConv}(in\_channels, out\_channels, edge\_dim=32)$$

---

## 6. Future Compatibility Roadmap
The `GeometryFeatures` container is designed to support future geometric expansions without breaking downstream consumers:
- **Relative Vectors**: $\vec{r}_{ij} = \vec{p}_i - \vec{p}_j$ (for equivariant updates).
- **Directions**: $\hat{r}_{ij} = \vec{r}_{ij} / d_{ij}$.
- **Edge Index**: Store connectivity mappings.
- **Metadata**: Dictionary storing precision types and parsing parameters.
