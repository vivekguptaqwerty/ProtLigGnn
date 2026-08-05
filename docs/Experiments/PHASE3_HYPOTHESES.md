# Phase 3.1A Scientific Hypotheses
**Milestone**: Phase 3.1A  
**Experiment ID**: P3.1A-GEOMETRY-LIGAND  
**Status**: Formalized  

This document formalizes the null and alternative hypotheses to be tested during the controlled benchmarking of **Phase 3.1A (Geometry Feature Injection)**.

---

## 1. Null Hypothesis ($H_0$)
- **Statement**: Adding explicit geometric edge representations (Euclidean distances mapped via Gaussian RBF) to the ligand encoder provides no statistically significant improvement in protein-ligand binding affinity prediction.
- **Mathematical Form**:
  $$\mu_{\text{RMSE}}^{\text{Geometry}} \ge \mu_{\text{RMSE}}^{\text{Baseline}}$$
  where $\mu_{\text{RMSE}}$ is the mean Root Mean Squared Error evaluated across the 5 canonical seeds.

---

## 2. Alternative Hypotheses

### Hypothesis 1: Representation Quality ($H_1$)
- **Statement**: Explicit geometric edge features improve the quality of graph representations, leading to lower prediction error.
- **Verification Metric**: Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).
- **Success Criteria**: A statistically significant reduction in mean RMSE across the 5 seeds ($p < 0.05$ on a paired t-test).

### Hypothesis 2: Mitigation of Regression-to-the-Mean ($H_2$)
- **Statement**: Explicit geometry provides structural anchors that allow the model to distinguish extreme binders, reducing regression-to-the-mean bias.
- **Verification Metric**: Slope of the residual vs. true affinity trendline.
- **Success Criteria**: An increase in the regression slope (moving closer to $0.0$, indicating reduced correlation between residuals and true affinities).

### Hypothesis 3: Pearson Correlation Improvement ($H_3$)
- **Statement**: Explicit distance features allow the model to capture relative binding strengths better, improving linear and rank correlation.
- **Verification Metric**: Pearson Correlation Coefficient (PCC) and Spearman Rank Correlation.
- **Success Criteria**: An increase in the mean PCC across the 5 seeds.

### Hypothesis 4: Interaction Localization ($H_4$)
- **Statement**: Incorporating explicit ligand geometry improves the localization of learned inter-graph interactions during cross-attention.
- **Verification Metric**: Pearson correlation between cross-attention weights and physical atom-residue distances.
- **Success Criteria**: An increase in the negative correlation between attention coefficients and physical distances (indicating higher attention weights are assigned to physically closer contacts).
