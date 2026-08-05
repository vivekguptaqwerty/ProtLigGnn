# Phase 3.1B Interaction Localization Report
**Experiment ID**: `P3.1B-GEOMETRY-ATTENTION`  
**Status**: Evaluation Completed  
**Date**: 2026-07-15  

This report evaluates whether injecting continuous physical 3D distance information directly into the cross-attention logits forces the attention mechanism to localize on close spatial contacts.

---

## 1. Attention-Distance Localization Metrics

We compute the Pearson/Spearman correlation between attention weights and inverse physical distances ($1/D$), the attention entropy (normalized), and the localization score (percentage of attention allocated to contacts $< 4.0\text{ Å}$) across the test set.

| Head | Pearson $r$ (vs $1/D$) | Spearman $\rho$ (vs $1/D$) | Entropy (Normalized) | Proximity Loc ($D < 4.0\text{ Å}$) |
| :---: | :---: | :---: | :---: | :---: |
| **Head 0** | $-0.0511$ | $-0.0218$ | $0.0621$ | $0.0050$ |
| **Head 1** | $+0.0044$ | $+0.0378$ | $0.0747$ | $0.0065$ |
| **Head 2** | $-0.0179$ | $+0.0159$ | $0.0618$ | $0.0066$ |
| **Head 3** | $-0.0714$ | $-0.0302$ | $0.0683$ | $0.0051$ |
| **Mean** | **$-0.0340$** | **$+0.0005$** | **$0.0667$** | **$0.0058$** |

### Interpretation
- **Spatial Alignment**: The correlation between attention weights and inverse physical distances remains extremely close to zero across all heads (mean Pearson $r = -0.0340$).
- **Entropy Focus**: The low normalized entropy (mean $0.0667$) indicates that the attention distribution is highly focused on a small subset of key nodes, but these nodes do not correspond to the closest physical contacts.
- **Why Localization remains weak**: The model is trained purely on the regression objective of binding affinity prediction. Without a direct contact-prediction loss or auxiliary distance-alignment loss, the GNN utilizes attention weights to route topological pocket node features that correlate with chemical affinity, rather than mimicking physical contact proximity.

---

## 2. Visualization of Attention Maps

The following figures illustrate the attention weights (Head 2) and pairwise physical distance matrices for representational test complexes.

### 2.1 Best Prediction (`3qj9`)
![Best prediction attention heatmap](file:///C:/Users/Harshhhh/.gemini/antigravity/brain/03c21182-9e56-4764-a6fd-83660d6d0fc1/attention_heatmap_best.png)

### 2.2 Median Prediction (`3ptg`)
![Median prediction attention heatmap](file:///C:/Users/Harshhhh/.gemini/antigravity/brain/03c21182-9e56-4764-a6fd-83660d6d0fc1/attention_heatmap_median.png)

### 2.3 Worst Prediction (`2xpk`)
![Worst prediction attention heatmap](file:///C:/Users/Harshhhh/.gemini/antigravity/brain/03c21182-9e56-4764-a6fd-83660d6d0fc1/attention_heatmap_worst.png)

The plots demonstrate that the attention maps do not align with physical close contacts (which appear as dark diagonal bands in the distance matrix), but are instead distributed across specific key topological residue channels.
