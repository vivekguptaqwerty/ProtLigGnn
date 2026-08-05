# Phase 3.1C Hypotheses Specification
**Experiment ID**: `P3.1C-EDGE-AWARE-INTERACTION`  
**Status**: Proposed  

This document specifies the scientific hypotheses for Phase 3.1C in rigorous statistical notation.

---

## 1. Regression Accuracy Hypotheses

### Hypothesis 0 ($H_0$)
Injecting multi-branch pairwise interaction representations (geometry, chemistry, and representation) does not improve regression performance compared to distance-biased attention.
$$\mu_{\text{RMSE}}^{\text{IRM}} \geq \mu_{\text{RMSE}}^{\text{DistanceBias}}$$

### Hypothesis 1 ($H_1$)
Injecting multi-branch pairwise interaction representations significantly reduces binding affinity prediction error.
$$\mu_{\text{RMSE}}^{\text{IRM}} < \mu_{\text{RMSE}}^{\text{DistanceBias}}$$

- **Statistical Test**: Paired two-sample t-test over test set RMSE across the 5 canonical seeds.
- **Rejection Criterion**: $p < 0.05$ and $\Delta \mu < 0$.

---

## 2. Interaction Alignment Hypotheses

### Hypothesis 2 ($H_0$)
The multi-branch interaction representations do not alter spatial attention localization or alignment.
$$\mu_{\text{PCC\_attention\_dist}}^{\text{IRM}} \leq \mu_{\text{PCC\_attention\_dist}}^{\text{DistanceBias}}$$

### Hypothesis 3 ($H_1$)
The multi-branch interaction representations improve physical contact localization and alignment.
$$\mu_{\text{PCC\_attention\_dist}}^{\text{IRM}} > \mu_{\text{PCC\_attention\_dist}}^{\text{DistanceBias}}$$

- **Statistical Test**: Two-sample t-test on attention-distance Pearson correlation coefficients.
