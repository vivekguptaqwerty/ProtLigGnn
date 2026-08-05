# Model Interpretability Report
## ProtLigGNN Optimized Baseline v1.1 Cross-Attention Analysis

**Target Model Version**: Optimized Baseline v1.1 (hidden_dim=128)  
**Analyzed Complexes**: Best (`2y7i`), Median (`3pvw`), and Worst (`2xpk`) prediction complexes  
**Analysis Date**: 2026-07-12  

---

## 1. Executive Summary

This report presents a rigorous interpretability analysis of the bidirectional cross-attention mechanism in **ProtLigGNN Optimized Baseline v1.1**. By extracting multi-head attention weights across three test complexes representing varying levels of prediction accuracy, we investigate whether the model's self-selected attention coordinates correlate with physical spatial distance.

---

## 2. Multi-Head Attention Head Heatmap

The figure below displays the attention weights for each of the 4 attention heads across the best, median, and worst predicted complexes:

![Attention Head Visualizations](file:///C:/Users/Harshhhh/.gemini/antigravity/brain/03c21182-9e56-4764-a6fd-83660d6d0fc1/attention_heads_multi_panel.png)

---

## 3. Quantitative Attention Entropy & Physical Distance Correlation

We extract the pairwise Euclidean distance matrix $D_{i, j} = \| \mathbf{pos}_{lig, i} - \mathbf{pos}_{prot, j} \|_2$ for all ligand heavy atoms $i$ and protein pocket residues $j$, and compute the Pearson correlation coefficient $r$ against the attention weight matrices $A_{h, i, j}$ for each head $h$.

| Complex | PDB ID | Head 1 | Head 2 | Head 3 | Head 4 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Best** (Error = 0.06) | **2y7i** | $H_{norm}=0.90$<br>$r = +0.17$ | $H_{norm}=0.78$<br>$r = +0.23$ | $H_{norm}=0.76$<br>$r = +0.13$ | $H_{norm}=0.74$<br>$r = \mathbf{-0.16}$ |
| **Median** (Error = 1.02) | **3pvw** | $H_{norm}=0.89$<br>$r = +0.17$ | $H_{norm}=0.87$<br>$r = +0.10$ | $H_{norm}=0.84$<br>$r = +0.08$ | $H_{norm}=0.83$<br>$r = +0.09$ |
| **Worst** (Error = 4.92) | **2xpk** | $H_{norm}=0.89$<br>$r = +0.13$ | $H_{norm}=0.63$<br>$r = +0.02$ | $H_{norm}=0.83$<br>$r = +0.03$ | $H_{norm}=0.82$<br>$r = -0.05$ |

*Note: $H_{norm}$ represents the mean attention entropy normalized by the maximum possible entropy ($\log P_{nodes}$). An $H_{norm} \approx 1.0$ indicates uniform, diffuse attention. Pearson $r$ p-values are all $< 10^{-5}$ unless otherwise indicated.*

---

## 4. Scientific Conclusions

### 4.1 Observations
1. **Diffuse Attention Distributions**: In all three complexes, the normalized attention entropy remains high ($H_{norm}$ ranging from $0.74$ to $0.90$ across most heads). This indicates that the GNN distributes attention broadly across pocket residues rather than focusing strictly on a single contact pair.
2. **Weak Distance Correlation**: The correlation between physical Euclidean distance and attention coefficients is weak ($|r| < 0.25$). For the best-predicted complex (`2y7i`), Head 4 exhibits a statistically significant negative correlation ($r = -0.16$, $p < 10^{-5}$), showing that it assigns slightly higher attention weights to physically closer atom-residue pairs. Other heads and complexes, however, exhibit weak positive correlations.

### 4.2 Hypotheses
- The high attention entropy and weak physical distance correlation suggest that the coordinate-free model relies on global topological patterns or pocket-wide sequence descriptors to predict binding affinity, rather than learning localized, physical atom-to-residue contact interfaces.

### 4.3 Limitations
- **No Causal Interpretation**: Attention weights indicate features that the GNN associates during training, but they **must not** be interpreted as causal physical binding forces or actual residue importance.
- **Topological Restriction**: Because the input graphs do not receive explicit 3D distances during message passing, any distance correlation is implicit and relies on structural graph connectivity as a proxy.

### 4.4 Future Work
- In Phase 3, we will investigate whether explicit coordinate representations (such as radial basis function distance bins or SE(3)-equivariant operations) can bias the model toward learning physically realistic binding coordinates.
