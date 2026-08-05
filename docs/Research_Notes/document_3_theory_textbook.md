# Document 3: Complete Theory and Working of the LigProtGNN Family

This document provides a textbook-style, first-principles derivation of the mathematical models, message-passing frameworks, and geometric attention equations defining the **LigProtGNN Research Suite**.

---

## 1. Graph Neural Networks & Message Passing

### 1.1 Message Passing Framework
A graph is defined as \(\mathcal{G} = (\mathcal{V}, \mathcal{E})\), where \(\mathcal{V}\) is the set of vertices (atoms or residues) and \(\mathcal{E}\) is the set of edges (covalent bonds or spatial proximity contacts). Node features are represented as \(h_i^{(0)} \in \mathbb{R}^{D_f}\).

In every GNN layer \(l\), message passing updates the latent representation of node \(i\) by aggregating messages from its immediate neighbors \(\mathcal{N}(i)\):
\[
m_i^{(l+1)} = \text{Aggregate}^{(l+1)} \left( \left\{ \text{Message}^{(l+1)}\left(h_i^{(l)}, h_j^{(l)}, e_{ij}\right) : j \in \mathcal{N}(i) \right\} \right)
\]
\[
h_i^{(l+1)} = \text{Update}^{(l+1)} \left( h_i^{(l)}, m_i^{(l+1)} \right)
\]
where:
- \(\text{Message}\) is a differentiable mapping (e.g. MLP).
- \(\text{Aggregate}\) is a permutation-invariant operator (e.g. sum, mean, or max).
- \(\text{Update}\) is a gating or projection sequence (e.g. GRU or Linear + LayerNorm).

### 1.2 Graph Attention Network (GATv2) Formulation
To model anisotropic communication, GATv2 introduces dynamic attention coefficients where the query depends non-linearly on the concatenated representations:
\[
e_{ij} = a^T \text{LeakyReLU}\left( W [h_i \| h_j] \right)
\]
\[
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}
\]
The updated feature is the weighted sum of projected neighbor representations:
\[
h_i^{(l+1)} = \sigma \left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij} W_v h_j \right)
\]

---

## 2. Geometric & Spatial Formulations

### 2.1 Coordinate Distances
Let \(p_i \in \mathbb{R}^3\) be the Cartesian coordinates of node \(i\). The Euclidean distance between ligand atom \(i\) and protein residue \(j\) is:
\[
d_{ij} = \|p_i - p_j\|_2 = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2 + (z_i - z_j)^2}
\]

### 2.2 Radial Basis Functions (RBF)
To map continuous distances into stable representations without gradient explosion, LigProtGNN applies Gaussian RBF kernels centered at intervals \(\mu_k\):
\[
\phi_k(d_{ij}) = \exp \left( -\gamma (d_{ij} - \mu_k)^2 \right)
\]
where:
- \(\mu_k\) are evenly spaced centers between a start parameter (e.g. 0.0 Å) and a cutoff parameter (e.g. 12.0 Å).
- \(\gamma\) controls the kernel width, typically set to:
  \[
  \gamma = \frac{1}{\Delta \mu^2}
  \]

---

## 3. Bidirectional Cross-Attention with Spatial Biases

Cross-attention allows node embeddings from the ligand (\(X_{lig}\)) to query node embeddings from the protein (\(X_{prot}\)) and vice versa. 

For the ligand-to-protein direction, query, key, and value matrices are computed:
\[
Q = X_{lig} W_q, \quad K = X_{prot} W_k, \quad V = X_{prot} W_v
\]
The spatial bias matrix \(B \in \mathbb{R}^{L \times P \times N_{heads}}\) acts as a masking logit modifier. The attention weights for head \(h\) are:
\[
A_h = \text{Softmax}\left( \frac{Q_h K_h^T}{\sqrt{d_k}} + \exp(\log\alpha_h) \cdot B_{h} \right)
\]
where:
- \(\alpha_h = \exp(\log\alpha_h)\) is a learnable scaling parameter in log-space.
- The attention context output is:
  \[
  \text{Context}_h = A_h V_h
  \]

---

## 4. Interaction Representation Module (IRM)

The IRM constructs a multidimensional representation matrix of shape \([L, P, D_{interaction}]\) using three parallel feature encoders:

```
                  +-----------------------------------+
                  |        InteractionContext         |
                  +-----------------+-----------------+
                                    |
            +-----------------------+-----------------------+
            |                       |                       |
+-----------v-----------+   +-------v-------+   +-----------v-----------+
|    Geometry Branch    |   |  Chem Branch  |   | Representation Branch |
|  - 32-dim Gaussian    |   |  - Element ID |   |  - GNN node latents   |
|    RBF distances      |   |  - properties |   |  - symmetric delta    |
+-----------+-----------+   +-------+-------+   +-----------+-----------+
            |                       |                       |
            +-----------------------+-----------------------+
                                    |
                        +-----------v-----------+
                        |     ConcatFusion      |
                        |  - project to latent  |
                        +-----------+-----------+
                                    |
                        +-----------v-----------+
                        |  Latent Space Flow    |
                        |  - Linear -> LN       |
                        |  - ReLU -> Dropout    |
                        +-----------+-----------+
                                    |
                        +-----------v-----------+
                        |    Logit Projector    |
                        |  - scale attention    |
                        +-----------------------+
```

### 4.1 Geometry Encoder
Converts Euclidean distances into a Gaussian-basis matrix:
\[
f_{geo}(i, j) = \text{RBF}(d_{ij}) \in \mathbb{R}^{32}
\]

### 4.2 Chemistry Encoder
Retrieves physical properties (atomic number, mass, electronegativity, covalent radius, period, group, common valence) for element types:
\[
v(i) = [z_i/100, m_i/250, \chi_i/4, \dots] \in \mathbb{R}^9
\]
The pairwise chemistry representation combines node features and the electronegativity gap \(\Delta \chi_{ij}\):
\[
f_{chem}(i, j) = [v_{lig}(i) \,\|\, v_{prot}(j) \,\|\, |\chi_i - \chi_j|] \in \mathbb{R}^{118}
\]

### 4.3 Representation Encoder
Combines GNN-derived node representations using a symmetric transformation:
\[
f_{rep}(i, j) = [h_i \,\|\, h_j \,\|\, |h_i - h_j| \,\|\, h_i \odot h_j]
\]

### 4.4 ConcatFusion & Projection Flow
The branches are concatenated along the feature dimension and projected:
\[
f_{fused}(i, j) = W_{fusion} [f_{geo}(i, j) \,\|\, f_{chem}(i, j) \,\|\, f_{rep}(i, j)] + b_{fusion}
\]
The fused matrix is processed sequentially:
\[
z_{latent} = \text{LayerNorm}(f_{fused})
\]
\[
z_{latent} = \text{ReLU}(z_{latent})
\]
\[
z_{interaction} = \text{Dropout}(z_{latent})
\]
This interaction representation is projected to calculate attention logits:
\[
\text{Bias}_{h}(i, j) = W_{bias\_proj} z_{interaction}
\]
