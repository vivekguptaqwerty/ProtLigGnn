# Phase 3.1B Scientific Hypotheses (Revised)
**Experiment ID**: `P3.1B-GEOMETRY-ATTENTION`  
**Certified Date**: 2026-07-15  

This document formalizes the hypotheses tested in the distance-biased cross-attention experiment.

---

## 1. Null Hypothesis ($H_0$)
- **Statement**: Injecting explicit Euclidean distance bias directly into the cross-attention interaction block produces no statistically significant difference in protein-ligand binding affinity prediction or attention localization compared to the certified `Optimized Baseline v1.1`.
- **Mathematical Form**:
  $$\mu_{\text{RMSE}}(\text{Geometry Attention}) = \mu_{\text{RMSE}}(\text{Baseline v1.1})$$
  $$\mu_{\text{MAE}}(\text{Geometry Attention}) = \mu_{\text{MAE}}(\text{Baseline v1.1})$$
  $$\mu_{\text{Localization}}(\text{Geometry Attention}) = \mu_{\text{Localization}}(\text{Baseline v1.1})$$

---

## 2. Alternative Hypotheses ($H_1, H_2, H_3, H_4$)

### Hypothesis 1: Binding Affinity Prediction Accuracy ($H_1$)
- **Statement**: Distance-biased cross-attention improves binding affinity prediction accuracy on the test set.
- **Verification Metric**: Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).
- **Goal**: Mean RMSE is reduced relative to Baseline v1.1 ($1.6048$) with a paired $p$-value $< 0.05$ across the 5 canonical seeds.

### Hypothesis 2: Reduction in Regression-to-the-Mean ($H_2$)
- **Statement**: Incorporating explicit spatial distance limits into the cross-attention logits reduces the model's systematic tendency to over-predict low affinities and under-predict high affinities.
- **Verification Metric**: Slope and correlation coefficient ($r$) of residuals vs. true affinities.
- **Goal**: Residual correlation with true affinity is weakened (closer to $0.0$), and the regression slope is shallower (closer to $0.0$, moving from baseline $-0.66$ towards $0.0$).

### Hypothesis 3: Linear & Monotonic Association ($H_3$)
- **Statement**: Introducing geometric constraints to interaction modeling improves the ranking capability and correlation of predicted values.
- **Verification Metric**: Pearson Correlation Coefficient (PCC) and Spearman Rank Correlation.
- **Goal**: Statistically significant relative improvement in mean PCC and mean Spearman.

### Hypothesis 4: Physical Spatial Attention Localization ($H_4$)
- **Statement**: Distance-biased cross-attention results in attention distributions that correlate strongly with physical contact distances.
- **Verification Metric**: Pearson and Spearman correlation between attention weights and inverse physical distances, and normalized attention entropy.
- **Goal**: Pearson correlation against inverse physical distance increases significantly ($r > 0.40$ on average) and attention entropy decreases, indicating focused spatial localization.
