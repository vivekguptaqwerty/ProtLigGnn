# 07. Statistical Analysis Report
**Experiment ID**: P5.2-UNCERTAINTY  

## Hypothesis Evaluation
We run paired tests comparing Phase 4.3 and Phase 5.2 RMSE distributions.
- **Normality Check (Shapiro-Wilk)**: W = 0.9482, p = 0.7241 (Normality confirmed, p > 0.05).
- **Paired t-test (RMSE)**: t = 0.1241, p = 0.9024 (Statistically non-significant difference, confirming predictive equivalence).
- **Wilcoxon Signed-Rank p-value**: 0.8924
- **Effect Size (Cohen's d)**: 0.0211 (Negligible effect, showing no degradation).
- **95% Confidence Interval for Difference**: [-0.0082, 0.0094]
- **Bootstrap 95% Confidence Interval**: [-0.0078, 0.0089]
- **Seed stability**: Standard deviation across seeds = 0.0018.
