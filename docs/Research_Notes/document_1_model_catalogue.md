# Document 1: LigProtGNN Family Model Catalogue & Technical Specifications

This register categorizes all architectures within the **LigProtGNN Research Suite**. It clearly demarcates:
1. **Section I: Implemented & Benchmarked Models** (Actual experimental runs and results).
2. **Section II: Planned Research Evolution Models** (Specifications, algorithms, and theory without fabricated metrics).
3. **Section III: External State-of-the-Art Baselines** (Published baseline profiles used for external comparison).

---

# SECTION I — IMPLEMENTED MODELS

## 1. ProtLigGNN Baseline v1.0
* **Theory**: Independent coordinate-free graph representations for protein and ligand structures. Standard spatial relationships are omitted; graph topologies are modeled via localized Graph Convolutional Network (GCN) layers.
* **Motivation**: Establishes the topological GNN baseline. Solves the limitation of traditional descriptors (FP/descriptor similarity) by learning features directly from graph topologies.
* **Architecture**: Two parallel, uncoupled graph encoder branches (protein and ligand) followed by global average pooling and a fully connected regressor.
* **Mathematics**:
  \[
  h_i^{(l+1)} = \sigma \left( D^{-1/2} A D^{-1/2} h_i^{(l)} W^{(l)} \right)
  \]
* **Tensor Flow**: 
  - Ligand: \(X_{lig\_raw} \in \mathbb{R}^{L \times 78} \to h_{lig} \in \mathbb{R}^{L \times 256} \to z_{lig} \in \mathbb{R}^{256}\)
  - Protein: \(X_{prot\_raw} \in \mathbb{R}^{P \times 30} \to h_{prot} \in \mathbb{R}^{P \times 256} \to z_{prot} \in \mathbb{R}^{256}\)
  - Regressor: \([z_{lig} \,\|\, z_{prot}] \in \mathbb{R}^{512} \to \hat{y} \in \mathbb{R}^1\)
* **Software Implementation**: Exists within [protliggnn_train.py](file:///d:/ProtLigGnn/protliggnn_train.py#L650-L750) under class `ProtLigGNN`.
* **Hyperparameters**: Latent dim = 256, dropout = 0.2, learning rate = 1e-3, optimizer = AdamW.
* **Benchmark Results**: Mean RMSE = \(1.5748 \pm 0.0592\).
* **Statistical Analysis**: Establishes the starting benchmark point.
* **Limitations**: Lacks geometric coordinate awareness and dynamic protein-ligand communication.
* **Promotion Decision**: Promoted (Historical baseline, superseded by v1.1).

## 2. Optimized Baseline v1.1
* **Theory**: Baseline v1.0 augmented with internal LayerNorm layers and optimized dropout rates to control covariate shift and mitigate overfitting.
* **Motivation**: Establishes a highly stable benchmark baseline. Resolves gradient degradation during backpropagation.
* **Architecture**: Adds LayerNorm after the GCN layers and adjusts regressor dropout to 0.2.
* **Mathematics**:
  \[
  z = \text{LayerNorm}(W h + b)
  \]
* **Tensor Flow**: Same as v1.0, with LayerNorm scaling applied prior to pooling.
* **Software Implementation**: Configured via command line option `--model_type baseline` in [protliggnn_train.py](file:///d:/ProtLigGnn/protliggnn_train.py#L1151).
* **Hyperparameters**: Latent dim = 256, dropout = 0.2, weight decay = 1e-4, learning rate = 1e-3.
* **Benchmark Results**: Mean RMSE = \(1.6048 \pm 0.0581\).
* **Statistical Analysis**: Serves as the primary reference point.
* **Limitations**: Lacks coordinate features and active cross-graph communication.
* **Promotion Decision**: Promoted (Certified Baseline).

## 3. Geometry v1.2 (Phase 3.1A)
* **Theory**: Integrates spatial distances mapped using Gaussian Radial Basis Functions (RBF) to inject spatial coordinate awareness.
* **Motivation**: Solves the limitation of coordinate-free GNN models.
* **Architecture**: Adds a pairwise distance calculator and Gaussian RBF mapping layer.
* **Mathematics**:
  \[
  \phi_k(d_{ij}) = \exp \left( -\gamma (d_{ij} - \mu_k)^2 \right)
  \]
* **Tensor Flow**:
  - Distance Matrix: \(D \in \mathbb{R}^{L \times P}\)
  - RBF Matrix: \(\Phi \in \mathbb{R}^{L \times P \times 32}\)
* **Software Implementation**: Handled in [models/geometry_attention/attention_bias.py](file:///d:/ProtLigGnn/models/geometry_attention/attention_bias.py) under the `RBFAttentionBias` module.
* **Hyperparameters**: RBF basis = 32, cutoff distance = 12.0 Å, gamma = 1.0.
* **Benchmark Results**: Mean RMSE = \(1.5443 \pm 0.0818\).
* **Statistical Analysis**: Relative RMSE reduction of **3.77%** over Baseline v1.1. Paired t-test: \(p = 0.1484\) (Non-significant).
* **Limitations**: Spatial distance features are static and do not influence GNN node embeddings directly during message passing.
* **Promotion Decision**: Promoted.

## 4. Geometry Attention v1.3 (Phase 3.1B)
* **Theory**: Implements bidirectional cross-attention biased by spatial distances, allowing node embeddings to update dynamically using coordinate signals.
* **Motivation**: Resolves the uncoupled representation limitation of v1.2.
* **Architecture**: Adds cross-attention layers using MultiheadAttention with distance bias masks.
* **Mathematics**:
  \[
  A = \text{Softmax}\left( \frac{QK^T}{\sqrt{d_k}} + \exp(\log\alpha) \cdot \text{Bias}_{RBF} \right)
  \]
* **Tensor Flow**:
  - Attention Logits: \(\mathbb{R}^{N_{heads} \times L \times P}\)
  - Updated Node Embeddings: \(X_{lig} \in \mathbb{R}^{L \times 256}\), \(X_{prot} \in \mathbb{R}^{P \times 256}\)
* **Software Implementation**: [models/geometry_attention/geometry_attention_model.py](file:///d:/ProtLigGnn/models/geometry_attention/geometry_attention_model.py).
* **Hyperparameters**: Attention heads = 4, initial alpha = 0.1 (learnable scale), cutoff = 12.0 Å.
* **Benchmark Results**: Mean RMSE = \(1.5529 \pm 0.0727\).
* **Statistical Analysis**: Paired t-test: \(p = 0.01778\) (Significant RMSE reduction over Baseline v1.1).
* **Limitations**: Attention bias uses only Euclidean distance, ignoring chemical and latent properties of atom pairs.
* **Promotion Decision**: Promoted (Active Baseline).

## 5. Interaction Representation Module v1.4 (Phase 3.1C Candidate)
* **Theory**: Upgrades static distance bias to a learnable, multi-branch pairwise interaction representation matrix.
* **Motivation**: Evaluates if explicit modeling of coordinate geometry, chemical properties, and node latents outperforms simple distance bias.
* **Architecture**: Adds parallel encoders (Geometry, Chemistry, Representation) fused via `ConcatFusion` and projected to attention logits.
* **Mathematics**:
  \[
  f_{chem}(i,j) = [v_{lig}(i) \,\|\, v_{prot}(j) \,\|\, |\chi_i - \chi_j|]
  \]
  \[
  z = \text{Dropout}(\text{ReLU}(\text{LayerNorm}(\text{Linear}(f_{fused}))))
  \]
* **Tensor Flow**:
  - Branches: \(\mathbb{R}^{L \times P \times 32}\)
  - Fused: \(\mathbb{R}^{L \times P \times 32}\)
  - Output Bias: \(\mathbb{R}^{N_{heads} \times L \times P}\)
* **Software Implementation**: [models/geometry_interaction/geometry_interaction_model.py](file:///d:/ProtLigGnn/models/geometry_interaction/geometry_interaction_model.py).
* **Hyperparameters**: Latent dim = 32, interaction dim = 32, heads = 4, dropout = 0.1.
* **Benchmark Results**: Mean RMSE = \(1.5535 \pm 0.1173\).
* **Statistical Analysis**: Non-significant compared to v1.3 (\(p = 0.9925\)).
* **Limitations**: Added complexity without statistical validation.
* **Promotion Decision**: **REJECTED** (v1.3 retained as active baseline).

---

# SECTION II — PLANNED RESEARCH EVOLUTION

## 1. Phase 3.2 — Physics-aware Interaction Module
* **Scientific Motivation**: Machine learning representations of protein-ligand interfaces often ignore fundamental thermodynamic and physical forces, leading to unphysical predictions on out-of-distribution (OOD) targets.
* **Hypothesis**: Incorporating explicit molecular potentials (electrostatics, van der Waals, hydrogen bonding) directly into the cross-attention layers constrains model representations to reflect realistic physical interactions.
* **Expected Architecture**: A parallel physics branch inside the IRM that calculates empirical forcefield terms using coordinates and atom properties, added directly to attention logits.
* **Mathematical Formulation**:
  - Electrostatic Potential (Coulomb's Law with distance-dependent dielectric):
    \[
    E_{elec}(i,j) = \frac{q_i q_j}{4\pi\epsilon_0 \epsilon(r_{ij}) r_{ij}}, \quad \epsilon(r_{ij}) = 4r_{ij}
    \]
  - Van der Waals (Lennard-Jones 12-6 potential):
    \[
    E_{vdW}(i,j) = 4\epsilon_{ij} \left[ \left(\frac{\sigma_{ij}}{r_{ij}}\right)^{12} - \left(\frac{\sigma_{ij}}{r_{ij}}\right)^6 \right]
    \]
* **Algorithm & Pseudocode**:
  ```python
  def compute_physics_potentials(coords_lig, coords_prot, charges_lig, charges_prot, vdw_eps, vdw_sig):
      # Compute pairwise distance matrix [L, P]
      dists = torch.cdist(coords_lig, coords_prot)
      
      # Electrostatic term
      q_prod = charges_lig.unsqueeze(1) * charges_prot.unsqueeze(0)
      dielectric = 4.0 * dists
      elec_pot = q_prod / (dielectric * dists + 1e-8)
      
      # vdW term
      sig_mat = 0.5 * (vdw_sig.unsqueeze(1) + vdw_sig.unsqueeze(0))
      eps_mat = torch.sqrt(vdw_eps.unsqueeze(1) * vdw_eps.unsqueeze(0))
      ratio = sig_mat / (dists + 1e-8)
      vdw_pot = 4.0 * eps_mat * (ratio**12 - ratio**6)
      
      return elec_pot + vdw_pot
  ```
* **Expected Computational Complexity**: \(\mathcal{O}(L \cdot P)\)
* **Integration**: Added as a static physics branch parallel to the `ChemistryFeatureEncoder`.

## 2. Phase 3.3 — Multi-scale Interaction Learning
* **Scientific Motivation**: Binding affinities are dictated both by atom-atom contacts and pocket-level/domain-level macro-conformations. Single-resolution graph architectures fail to capture these cross-scale hierarchies.
* **Hypothesis**: Co-embedding residue-level protein graphs alongside atom-level representations using hierarchical pooling enables multi-scale feature propagation.
* **Expected Architecture**: Dual protein encoders (atom-level and residue-level) fused via cross-scale cross-attention.
* **Expected Computational Complexity**: \(\mathcal{O}(V_{atom} + V_{residue})\)

## 3. Phase 3.4 — Dynamic Protein-Ligand Modeling
* **Scientific Motivation**: Protein-ligand systems are highly dynamic ensembles rather than static lock-and-key coordinates.
* **Hypothesis**: Modeling ensembles of conformers using attention-based pooling captures the conformational entropy of the binding process.
* **Expected Architecture**: Encoders process multiple conformer coordinate batches, aggregated using self-attention:
  \[
  z_{joint} = \sum_{k} \beta_k z_{conformer\_k}
  \]

## 4. Phase 3.5 — Uncertainty Quantification
* **Scientific Motivation**: Standard regression heads produce point predictions without confidence boundaries, limiting their utility in drug discovery.
* **Hypothesis**: Evidential Deep Regression allows the model to predict both the target binding affinity and the underlying aleatoric and epistemic uncertainty parameters.
* **Expected Architecture**: Regressor head outputs four parameters (\(\gamma, \nu, \alpha, \beta\)) mapping a Normal-Inverse-Gamma distribution.

## 5. Phase 4.1 — SE(3)-Equivariant Interaction Network
* **Scientific Motivation**: Cartesian coordinates are sensitive to rotation and translation. GNN attention bias is invariant, but message passing should be equivariant to maintain spatial directions.
* **Hypothesis**: Updating coordinates using equivariant message passing (e.g. EGNN) preserves spatial geometric equivariance.
* **Mathematical Formulation**:
  \[
  m_{ij} = \phi_m(h_i, h_j, \|p_i - p_j\|^2)
  \]
  \[
  p_i^{(l+1)} = p_i^{(l)} + \sum_{j \in \mathcal{N}(i)} (p_i^{(l)} - p_j^{(l)}) \phi_x(m_{ij})
  \]

## 6. Phase 4.2 — Foundation Model Integration
* **Scientific Motivation**: GNN models trained on limited affinity datasets are prone to overfitting. Large protein language models (like ESM-2) contain deep evolutionary representations.
* **Hypothesis**: Initializing GNN protein nodes with ESM-2 embeddings transfers evolutionary priors, improving affinity predictions.

## 7. Phase 4.3 — Self-Supervised Pretraining
* **Scientific Motivation**: Label scarcity in binding affinity datasets restricts representations.
* **Hypothesis**: Pre-training the GNN on millions of unlabelled chemical and residue graphs using contrastive objectives (GraphCL) learns robust molecular representations.

## 8. Phase 4.4 — Diffusion-based Interaction Modeling
* **Scientific Motivation**: Standard models evaluate coordinates but cannot generate or adapt them.
* **Hypothesis**: Diffusion objectives denoise coordinate ensembles, allowing docking pose generation.

## 9. Phase 4.5 — Explainability Framework
* **Scientific Motivation**: Trustworthy deployment requires transparent model predictions.
* **Hypothesis**: Using Integrated Gradients over the spatial coordinates maps localized contribution features.

---

# SECTION III — FINAL UNIFIED LIGPROTGNN-X

The **LigProtGNN-X** system represents the complete, unified target architecture integrating all modular phases:

```
[Atom/Residue Features] ---> [ESM-2/ProtT5 Encoders] ---> [SE(3)-Equivariant GNN]
                                                                  |
[Conformer Coordinates] ---> [Ensemble Pooling] --------> [Cross-Scale Attention]
                                                                  |
[Physical Potentials] -----> [Physics logit bias] ------> [Dynamic IRM Bias]
                                                                  |
                                                          [Evidential Regressor]
                                                                  |
                                                      [Prediction & Uncertainty]
```
- **Equivariance**: Processes 3D coordinates using SE(3)-equivariant EGNN blocks.
- **Physics**: Employs forcefield potentials (electrostatics, vdW) as logit constraints.
- **Uncertainty**: Evidential regression output head.
- **Deployment**: Supports industrial inference via ONNX exports.
