# Interaction Latent Space & Advanced Metrics Specification
**Status**: Proposed  
**Version**: 2.0  

This document specifies the advanced metrics, visualization hooks, and analytical methods designed for the Interaction Latent Space.

---

## 1. Reusable Latent Analysis Hooks

The `Interaction Embedding` tensor $\mathbf{Z} \in \mathbb{R}^{L \times P \times D_{\text{int}}}$ can be intercepted at test time to generate publication-quality scientific metrics and figures:

- **Interaction Embedding Norm**: Measures the magnitude of interaction features across the interface:
  $$\|\mathbf{Z}\|_F = \sqrt{\sum_{i=1}^L \sum_{j=1}^P \|\mathbf{z}_{ij}\|_2^2}$$
- **Branch Contribution**: Variance-based attribution showing how much geometry, chemistry, and representation branches contribute to the latent variance.
- **Dimensionality Reduction**: Automatically runs UMAP/PCA on the test set interaction embeddings to cluster binding pockets.

---

## 2. Advanced Spatial Localization Metrics

Beyond simple Pearson/Spearman correlations, the evaluation pipeline computes:

1. **Top-K Localization Precision**:
   $$P@K = \frac{1}{K} \sum_{k=1}^K \mathbb{I}(d(i_k, j_k) < 4.0\text{ Å})$$
   measures if the top $K$ highest attention weights match actual physical contacts.
2. **Nearest Contact Recall**:
   $$\text{Recall} = \frac{\sum_{(i,j) \in C} \mathbb{I}(\text{attn}(i,j) > \text{threshold})}{|C|}$$
   where $C$ is the set of contacts $< 4.0\text{ Å}$.
3. **Mean Reciprocal Rank (MRR)**:
   $$\text{MRR} = \frac{1}{|C|} \sum_{(i,j) \in C} \frac{1}{\text{rank}(\text{attn}(i,j))}$$
4. **Distance Calibration Error (DCE)**:
   Measures how well attention probabilities calibrate with actual physical distances:
   $$\text{DCE} = \sum_{b=1}^B \frac{|V_b|}{N} |\bar{d}_b - P(\text{contact}_b)|$$
5. **Head Diversity**:
   $$\text{Div} = \frac{2}{H(H-1)} \sum_{h_1 < h_2} (1 - \text{cosine\_similarity}(\mathbf{A}_{h_1}, \mathbf{A}_{h_2}))$$
