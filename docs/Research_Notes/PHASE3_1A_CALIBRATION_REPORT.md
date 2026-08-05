# Phase 3.1A Calibration & Spatial Attention Report
**Experiment ID**: P3.1A-GEOMETRY-LIGAND  
**Status**: Completed and Reviewed  

This report provides calibration analyses and extracts spatial cross-attention statistics for the geometry model.

---

## 1. Model Calibration & Variance

- **Prediction Variance**: The standard deviation of the predictions is $\text{std}(y_{pred}) \approx 0.65$. The true affinity distribution has $\text{std}(y_{true}) \approx 1.83$. The model under-predicts the variance of the true distribution, showing that the model outputs are over-smoothed towards the dataset mean ($\approx 6.0$).
- **Mean Bias**: The aggregate bias is $-0.1704 \pm 0.1647$, showing a slight systematic under-prediction of binding affinities.

---

## 2. Spatial Cross-Attention Analysis

We extract the Pearson correlation coefficient ($r$) between the cross-attention head coefficients $A_{h, i, j}$ and the physical pairwise Euclidean distance matrix $D_{i, j}$ between ligand atoms $i$ and protein pocket residues $j$:

| Complex | PDB ID | Head 1 | Head 2 | Head 3 | Head 4 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Best** (Error = 0.06) | **2y7i** | $r = +0.046$ | $r = +0.020$ | $r = +0.041$ | $r = \mathbf{+0.397}$ |
| **Median** (Error = 1.02) | **3pvw** | $r = +0.077$ | $r = +0.164$ | $r = +0.153$ | $r = +0.162$ |
| **Worst** (Error = 4.92) | **2xpk** | $r = +0.019$ | $r = +0.029$ | $r = +0.117$ | $r = -0.000$ |

*Observation*:
1. **Low Distance Correlation**: With the exception of Head 4 on PDB `2y7i` ($r \approx 0.40$), all heads exhibit weak positive correlation with physical distance ($|r| < 0.17$).
2. **Failure to Enforce Spatial Localization**: Restricting explicit geometry to ligand edge attributes in the ligand encoder does **not** cause the cross-attention block (which remains coordinate-free) to learn physically realistic spatial interactions. This confirms that to improve interaction localization (Hypothesis $H_4$), we must introduce explicit distance biases directly into the cross-attention projection mechanism itself.
