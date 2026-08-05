# Document 5: External State-of-the-Art Comparison Chapter

This document provides a comprehensive analysis of external baseline architectures from the literature. It serves as a benchmark reference chapter, comparing the proposed **LigProtGNN-X** system against key molecular and protein representation frameworks.

---

## 1. Sequence-Based Representation Models

### 1.1 ESM-2 & ESM-3 (Evolutionary Scale Modeling)
* **Core Idea**: Transformer protein language models trained on billions of evolutionary sequences using Masked Language Modeling (MLM). ESM-3 expands this to structure and functional tokens.
* **Mathematical Formulation**:
  \[
  \mathcal{L}_{MLM} = -\sum_{i \in M} \log P(x_i \mid x_{\setminus M})
  \]
* **Strengths**: Learns deep evolutionary and structural priors directly from sequences.
* **Weaknesses**: Ignores explicit ligand geometric coordinates and binding-pocket physics.
* **Expected Comparison with LigProtGNN-X**: LigProtGNN-X uses ESM-2 embeddings to initialize protein node features but improves on it by executing 3D coordinate cross-attention.

### 1.2 ProtT5
* **Core Idea**: Sequence-to-sequence T5 model pre-trained on UniRef50 sequences.
* **Strengths**: Robust residue-level representations for general protein properties.
* **Weaknesses**: Topological only; lacks 3D structure and pocket binding awareness.

### 1.3 MolFormer
* **Core Idea**: Linear attention-based transformer trained on SMILES strings.
* **Strengths**: Learns robust global chemical motifs from massive SMILES collections.
* **Weaknesses**: Lacks 3D coordinates and structural binding conformations.

---

## 2. 3D Geometric & Docking Models

### 2.1 Uni-Mol
* **Core Idea**: A 3D spatial coordinate molecular framework pre-trained on conformation ensembles using masked coordinate denoising.
* **Mathematical Formulation**:
  \[
  \mathcal{L}_{coord} = \|p_{pred} - p_{true}\|_2^2
  \]
* **Strengths**: Deep coordinate awareness and robust SMILES-free atom latents.
* **Weaknesses**: Fixed conformations; does not model dynamic coordinate variance during docking.
* **Expected Comparison with LigProtGNN-X**: LigProtGNN-X integrates coordinate ensembles, allowing dynamic variance mapping during training.

### 2.2 DiffDock
* **Core Idea**: Generative diffusion model defining docking as a score-matching process over translation, rotation, and torsion keys.
* **Strengths**: State-of-the-art binding pose generation.
* **Weaknesses**: High latency; computationally expensive for high-throughput screening.

### 2.3 EquiBind
* **Core Idea**: An SE(3)-equivariant docking model mapping rigid coordinate translations and rotations.
* **Strengths**: Invariant to reference coordinate changes.
* **Weaknesses**: Struggles with flexible pocket conformations.

---

## 3. Structural Graph Neural Networks

### 3.1 GearNet
* **Core Idea**: Multi-relational GNN modeling structural residue networks.
* **Strengths**: Excellent structural protein representation.
* **Weaknesses**: Ignores atomic-level ligand details.

### 3.2 GIGN (Geometry Interaction Graph Network)
* **Core Idea**: Models binding interfaces using distance-cutoff contact graphs.
* **Strengths**: Highly efficient for interface modeling.
* **Weaknesses**: Lacks language model priors and evidential uncertainty bounds.
* **Expected Comparison with LigProtGNN-X**: LigProtGNN-X adds ESM-2 priors, evidential uncertainty regression, and coordinate-ensemble aggregation, yielding superior generalization.
