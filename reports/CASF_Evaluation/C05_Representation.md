# C05 Representation Quality: CASF-2016

## Representation Bottlenecks
- **Feature Redundancy**: Correlation analysis shows that GNN hidden activations are highly correlated, indicating representational collapse in hidden space.
- **Bidirectional Attention Projectors**: High projection dimensions (256) inside cross-attention layers suffer from feature under-determination under a small dataset.
- **Verdict**: Encoders are under-regularized. Adding LayerNorm in GCN layers and dropout on attention weights will reduce feature redundancy.
