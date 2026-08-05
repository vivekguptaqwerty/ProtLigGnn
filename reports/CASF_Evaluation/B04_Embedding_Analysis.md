# B04 Embedding Analysis: CASF-2016

## Principal Component Analysis (PCA)
- Joint pooling GNN representations (1024 dimensions) were extracted for PDBbind cache structures and projected onto 2 principal components.
- **PC 1 Explained Variance**: 37.0%
- **PC 2 Explained Variance**: 19.3%
- **Total Variance Explained (PC1 + PC2)**: 56.3%

## Correlation with Binding Affinity
- **PC 1 vs Affinity Pearson**: `0.1591`
- **PC 2 vs Affinity Pearson**: `-0.3103`

## Cluster Visualization
![Embedding PCA](embedding_pca.png)

*The PCA projection shows that GNN embeddings do correlate with binding affinity (PC2 exhibits a Pearson R of -0.31), but there is no clean geometric clustering or ordering, indicating that structural features are poorly mapped to the target regression space due to lack of training data.*
