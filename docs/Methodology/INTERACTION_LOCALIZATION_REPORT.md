# Phase 3.1B Interaction Localization Report
**Experiment ID**: `P3.1B-GEOMETRY-ATTENTION`  
**Status**: Template & Specification  

---

## 1. Metric Definitions

This report evaluates whether the distance-biased cross-attention layer forces the model to focus attention on physically close ligand-protein contacts.

### 1.1 Inverse Distance Correlation (Pearson & Spearman)
We compute the correlation coefficient between the attention weight matrix $A_{h, i, j}$ for head $h$ and the inverse pairwise Euclidean distance matrix $D_{ij}^{-1}$:
$$r_{\text{Pearson}} = \text{Corr}\left(\text{vec}(A_h), \text{vec}\left(D^{-1}\right)\right)$$
- **Expected Interpretation**: A high positive correlation ($r > 0.40$) indicates that atoms and residues that are physically close receive systematically higher attention weights.

### 1.2 Attention Entropy (Per-Head & Mean)
We compute the normalized entropy of the attention distribution across the query-key pairs:
$$\text{Entropy}(A_h) = -\frac{1}{L \cdot P} \sum_{i=1}^L \sum_{j=1}^P A_{h, i, j} \log\left(A_{h, i, j} + \epsilon\right)$$
- **Expected Interpretation**: Lower entropy indicates that the attention distribution is highly focused on specific key contacts rather than spread out uniformly.

---

## 2. Interaction Localization Table

| Head | Pearson $r$ (vs $D^{-1}$) | Spearman $\rho$ (vs $D^{-1}$) | Entropy (Normalized) | Localization Score ($D < 4.0\text{ Å}$) |
| :---: | :---: | :---: | :---: | :---: |
| **Head 1** | *Pending* | *Pending* | *Pending* | *Pending* |
| **Head 2** | *Pending* | *Pending* | *Pending* | *Pending* |
| **Head 3** | *Pending* | *Pending* | *Pending* | *Pending* |
| **Head 4** | *Pending* | *Pending* | *Pending* | *Pending* |
| **Mean** | *Pending* | *Pending* | *Pending* | *Pending* |

---

## 3. Distribution Visualizations
All future benchmark runs will plot and save:
1. **Histogram**: Shows the frequency distribution of attention weights compared to physical distances.
2. **Violin & Box Plots**: Shows the spread of head-wise localization scores across the test set.
