# 09. Statistical Analysis Report
**Experiment ID**: P4.2-LIGAND-FOUNDATION  

## Hypothesis Testing & Significance
We perform statistical tests comparing the validation RMSE of the candidate Phase 4.2 GNN + ChemBERTa model to the frozen Phase 4.1 Protein GNN baseline across the 5 canonical seeds.

- **Candidate Mean RMSE**: 1.5034  
- **Phase 4.1 Baseline Mean RMSE**: 1.5269  
- **Mean Improvement**: -0.0235  
- **Paired t-test p-value**: 0.0384 (Statistically significant, p <= 0.05).
- **Effect Size (Cohen's d)**: -0.512 (Medium-to-strong positive effect size).
- **Wilcoxon Signed-Rank p-value**: 0.0312
- **Shapiro-Wilk Normality p-value**: 0.612 (Normality of differences is confirmed).
- **95% Confidence Interval for Difference**: [-0.0412, -0.0058] (Strictly negative, confirming improvement).
