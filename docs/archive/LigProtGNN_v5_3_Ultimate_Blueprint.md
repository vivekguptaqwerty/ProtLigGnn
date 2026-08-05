# LigProtGNN-X v5.3: Next-Generation Binding Affinity Prediction Framework
## Central Novelty: Dynamic Interaction Memory Graph (DIMG) with Geometry-Aware Attention and Hybrid Physics Edge Representations

---

## 1. Formalization of Novelty: Dynamic Interaction Memory Graph (DIMG)

To elevate the scientific framing, we define the **Dynamic Interaction Memory Graph (DIMG)**:
> **Dynamic Interaction Memory Graph (DIMG)** is a rotation-equivariant, translation-equivariant, and topologically adaptive graph neural network representation. It updates binding pocket geometries and edge attributes dynamically during message passing while retaining structural context through a gated memory pathway.

All computational GNN components are framed as supporting infrastructure for DIMG.

---

## 2. Equivariance Theorem & Proof

### Theorem 1 ($SE(3)$ Equivariance of DIMG)
*Let $\mathcal{T}_{g}$ represent an $SE(3)$ transformation (composed of a rotation $R \in SO(3)$ and a translation $t \in \mathbb{R}^3$) applied to the 3D coordinate tensor $x \in \mathbb{R}^{N \times 3}$. The Dynamic Interaction Memory Graph coordinate updates $x^{l+1}$ and node features $h^{l+1}$ satisfy:*

$$h_i^{l+1}(\mathcal{T}_g(x), h^l) = h_i^{l+1}(x, h^l) \quad \text{(Invariance of features)}$$

$$x_i^{l+1}(\mathcal{T}_g(x), h^l) = R x_i^{l+1}(x, h^l) + t \quad \text{(Equivariance of coordinates)}$$

### Proof
1. The adaptive radius $r_i$ is computed from node features $h_i$ which are coordinate-independent. Thus, $r_i$ is $SE(3)$ invariant.
2. The dynamic graph edges $\mathcal{E}^{l}$ are rewired based on pairwise Euclidean distances $\|x_i - x_j\|$. Since $\|(R x_i + t) - (R x_j + t)\| = \|R(x_i - x_j)\| = \|x_i - x_j\|$, the distance is $SE(3)$ invariant. Consequently, the edge set $\mathcal{E}^{l}$ is invariant under $SE(3)$ transformations.
3. In the message passing step, messages $m_{ij}$ depend on invariant features $h_i, h_j$ and the invariant distance $d_{ij}$. Thus, $m_{ij}$ is $SE(3)$ invariant.
4. The coordinate update is given by:
   $$x_i^{l+1} = x_i^l + \sum_{j \in \mathcal{N}(i)} (x_i^l - x_j^l) \phi_x(m_{ij})$$
   Under transformation $\mathcal{T}_g$:
   $$x_i^{l+1}(\mathcal{T}_g(x)) = (R x_i^l + t) + \sum_{j \in \mathcal{N}(i)} ((R x_i^l + t) - (R x_j^l + t)) \phi_x(m_{ij})$$
   $$= (R x_i^l + t) + R \sum_{j \in \mathcal{N}(i)} (x_i^l - x_j^l) \phi_x(m_{ij})$$
   $$= R \left( x_i^l + \sum_{j \in \mathcal{N}(i)} (x_i^l - x_j^l) \phi_x(m_{ij}) \right) + t = R x_i^{l+1}(x) + t$$
   This proves coordinate equivariance.
5. Feature updates depend only on $h_i$ and messages $m_{ij}$, which are both invariant, preserving feature invariance. $\blacksquare$

---

## 3. Computational & Memory Complexity

| Module | Computational Complexity | Memory Complexity | Parameter Footprint |
| :--- | :--- | :--- | :--- |
| **Protein/Ligand EGNN** | $\mathcal{O}(E)$ | $\mathcal{O}(V + E)$ | ~1.2M |
| **Dynamic Graph Rewiring** | $\mathcal{O}(V \log V + E)$ | $\mathcal{O}(V + E)$ | ~0.1M |
| **Gated Edge Memory** | $\mathcal{O}(E)$ | $\mathcal{O}(E)$ | ~0.8M |
| **Geometry-Aware Attention** | $\mathcal{O}(k(V_p + V_l))$ | $\mathcal{O}(V_p + V_l)$ | ~0.5M |

*Where $V$ represents nodes, $E$ represents edges, and $k$ represents sparse kNN neighbor bounds.*

---

## 4. Gated Edge Transformer (Alternate to GRU)

To model interactions between adjacent edges (e.g., angle bend constraints), we compare the independent GRU memory update with a **Gated Edge Transformer** layer:

$$e_{ij}^l = \text{EdgeTransformer}(e_{ij}^{l-1}) + W_{msg} m_{ij}$$

where edge states query neighboring edge vectors sharing a common node, allowing edge-edge interaction.

---

## 5. Failure Mode & Robustness Analysis

### 1. Localized Failure Modes
- **Metal Complexes**: DIMG represents coordinate interactions, but coordinate covalent metal-ligand bonds require customized valence descriptors.
- **Membrane Proteins**: Hydrophobic residues in transmembrane channels show distribution shift relative to soluble PDBbind templates.
- **Weak Binders**: Evidential regression outputs large epistemic uncertainty $\text{Var}[\mu]$ when predicting affinities pK$_d < 3.0$.

### 2. Robustness Augmentation
To ensure generalization, training is augmented with:
- **Coordinate noise**: Adding Gaussian perturbation $\sigma = 0.05$Å to conformer coordinates.
- **Rotation**: Applying random $SO(3)$ rotation matrices to conformers during batch generation.
