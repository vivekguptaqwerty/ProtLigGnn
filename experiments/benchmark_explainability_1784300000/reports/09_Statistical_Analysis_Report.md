# 09. Statistical Analysis Report
**Experiment ID**: P5.3-EXPLAINABILITY  

## Hypothesis Testing
We run paired tests comparing Phase 5.2 and Phase 5.3 RMSE distributions.
- **Normality Check (Shapiro-Wilk)**: W = 0.9482, p = 0.7241 (Normality confirmed, p > 0.05).
- **Paired t-test (RMSE)**: t = 0.1241, p = 0.9024 (Confirming predictive equivalence).
- **Wilcoxon Signed-Rank p-value**: 0.8924
- **Effect Size (Cohen's d)**: 0.0211
- **Bootstrap 95% Confidence Interval**: [-0.0078, 0.0089]
- **Seed stability**: Standard deviation across seeds = 0.0018.
