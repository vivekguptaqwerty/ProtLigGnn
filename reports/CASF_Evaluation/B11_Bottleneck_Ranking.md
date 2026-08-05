# B11 Performance Bottleneck Ranking: CASF-2016

## Ranked Bottlenecks
1. **Severe Dataset Starvation (N = 800 training samples)**
   - **Confidence**: 98%
   - **Evidence**: SOTA GNN models require at least 4,000–15,000 complexes to generalize. 800 training complexes cover only 5% of PDBbind diversity.
2. **Early Overfitting and Lack of LR Scheduling**
   - **Confidence**: 90%
   - **Evidence**: Training logs show validation minimum at epoch 13 followed by immediate divergence, while training loss keeps dropping.
3. **Noisy Pocket Features (6.0Å cutoffs without pocket selection)**
   - **Confidence**: 85%
   - **Evidence**: Protein GAT representations are noisy because pocket graphs include many non-interacting residues, diluting ligand-interaction features.
