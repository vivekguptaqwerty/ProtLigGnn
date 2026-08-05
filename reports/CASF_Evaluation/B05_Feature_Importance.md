# B05 Feature Importance: CASF-2016

## Component Contributions
1. **Ligand Encoder (GCN)**: **High Contribution**. The GCN maps atomic features and coordinate geometry, providing key chemical descriptors.
2. **Protein Encoder (GAT)**: **Moderate Contribution**. GAT attention weights map binding pocket pocket contacts, but because pocket graphs are large, the GAT representation is noisy.
3. **Bidirectional Cross-Attention**: **Low Contribution**. The cross-attention projetion layers are highly parameterized and suffer from parameter starvation under the small training dataset.
