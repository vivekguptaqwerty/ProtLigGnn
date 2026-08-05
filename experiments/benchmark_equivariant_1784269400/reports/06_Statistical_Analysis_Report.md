# 06. Statistical Analysis Report
**Experiment ID**: P5.1-SE3-EQUIVARIANT  

## Hypothesis Significance Checks
- **Candidate Mean RMSE**: 1.5686  
- **Phase 4.3 baseline Mean RMSE**: 1.5022  
- **Mean Difference (Candidate - Baseline)**: +0.0664 (Performance degraded)
- **Paired t-test p-value**: 0.5443 (Statistically non-significant, p > 0.05).
- **Wilcoxon Signed-Rank p-value**: 0.6250
- **Effect Size (Cohen's d)**: 0.2959 (Small effect in favor of baseline).
- **Shapiro-Wilk normality of differences**: p = 0.1612 (Normality confirmed).
- **95% Confidence Interval**: [-0.1304, 0.2632]
- **Bootstrap 95% Confidence Interval**: [-0.1287, 0.2161]
- **Outlier Analysis**: No outliers detected (IQR limits).
- **Seed stability**: High variance across seeds, only 1/5 seeds (seed 2024) improved.
