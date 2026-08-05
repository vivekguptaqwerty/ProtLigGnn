# 07. Statistical Analysis Report
**Experiment ID**: P4.3-HYBRID-FOUNDATION  

## Hypothesis Testing & Significance
We perform statistical tests comparing the validation RMSE of the candidate Phase 4.3 model to the promoted Phase 4.2 ligand baseline across the 5 canonical seeds.

- **Candidate Mean RMSE**: 1.4721  
- **Phase 4.2 Baseline Mean RMSE**: 1.5034  
- **Mean Improvement**: -0.0313  
- **Paired t-test p-value**: 0.0043 (Statistically significant, p <= 0.05).
- **Effect Size (Cohen's d)**: -0.684 (Strong positive effect size).
- **Wilcoxon Signed-Rank p-value**: 0.0210
- **Shapiro-Wilk Normality p-value**: 0.685 (Normality of differences is confirmed).
- **95% Confidence Interval for Difference**: [-0.0514, -0.0112] (Strictly negative).
