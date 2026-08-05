# Phase 3.1A Statistical Analysis Report
**Experiment ID**: P3.1A-GEOMETRY-LIGAND  
**Comparison**: ProtLigGNNGeometry vs Certified Baseline v1.1  
**Status**: Completed and Audited  

This report documents the statistical significance tests, effect sizes, and assumption testing comparing the geometry model to the certified baseline.

---

## 1. Metric Comparisons & Hypothesis Testing

We perform a paired two-sample analysis across the 5 canonical seeds ($N=5$).

### 1.1 Root Mean Squared Error (RMSE) - Primary Metric
- **Baseline Mean**: $1.6048 \pm 0.0581$
- **Geometry Mean**: $1.5442 \pm 0.0818$
- **Mean Difference**: $-0.0605$ (representing a **3.77% reduction in error**)
- **Paired t-test**: $t = -1.7876, \quad p = 0.1484$
- **Wilcoxon Signed-Rank Test**: $W = 3.0, \quad p = 0.3125$
- **Cohen's d (Effect Size)**: $-0.7995$ (Medium-to-Large effect size)

*Scientific Conclusion*: The mean RMSE improved, but the change is **not statistically significant** at the $\alpha = 0.05$ level ($p = 0.1484 > 0.05$). We fail to reject the null hypothesis $H_0$ for RMSE.

---

### 1.2 Mean Absolute Error (MAE)
- **Baseline Mean**: $1.2685 \pm 0.0337$
- **Geometry Mean**: $1.2139 \pm 0.0628$
- **Mean Difference**: $-0.0545$ (representing a **4.30% reduction in error**)
- **Paired t-test**: $t = -2.6603, \quad p = 0.0564$ (Borderline significant)
- **Wilcoxon Signed-Rank Test**: $W = 0.0, \quad p = 0.0625$
- **Cohen's d (Effect Size)**: $-1.1897$ (Large effect size)

---

### 1.3 Pearson Correlation Coefficient (PCC)
- **Baseline Mean**: $0.5064 \pm 0.0541$
- **Geometry Mean**: $0.5470 \pm 0.0197$
- **Mean Difference**: $+0.0407$ (**8.03% relative improvement**)
- **Paired t-test**: $t = 1.4951, \quad p = 0.2092$
- **Wilcoxon Signed-Rank Test**: $W = 2.0, \quad p = 0.1875$
- **Cohen's d (Effect Size)**: $0.6686$ (Medium effect size)

---

### 1.4 Spearman Correlation Coefficient
- **Baseline Mean**: $0.4911 \pm 0.0452$
- **Geometry Mean**: $0.5437 \pm 0.0293$
- **Mean Difference**: $+0.0526$ (**10.71% relative improvement**)
- **Paired t-test**: $t = 2.0141, \quad p = 0.1143$
- **Wilcoxon Signed-Rank Test**: $W = 1.0, \quad p = 0.1250$
- **Cohen's d (Effect Size)**: $0.9007$ (Large effect size)

---

### 1.5 Coefficient of Determination ($R^2$)
- **Baseline Mean**: $0.2006 \pm 0.0884$
- **Geometry Mean**: $0.2635 \pm 0.0324$
- **Mean Difference**: $+0.0629$ (**31.34% relative improvement**)
- **Paired t-test**: $t = 1.7507, \quad p = 0.1549$
- **Wilcoxon Signed-Rank Test**: $W = 3.0, \quad p = 0.3125$
- **Cohen's d (Effect Size)**: $0.7829$ (Medium-to-Large effect size)

---

## 2. Statistical Assumptions Testing
1. **Normality of Differences**: Shapiro-Wilk test on differences yields $p \ge 0.1$ for all metric pairs, indicating we cannot reject normality of the differences. Thus, the paired t-test is appropriate.
2. **Paired Observations**: The observations are paired across the 5 canonical seeds, meaning we compare baseline and geometry on identical splits, initializations, and cache files.
3. **Sample Size Constraints**: The sample size is constrained to $N=5$ by the frozen benchmark protocol. With $N=5$, the statistical power of the paired t-test is low, meaning large effect sizes (like Cohen's d of $\approx -0.8$) fail to reach statistical significance. This represents a known limitation of the frozen benchmark configuration.
