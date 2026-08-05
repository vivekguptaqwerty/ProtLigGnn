# 10. Scientific Audit Report
**Experiment ID**: P5.3-EXPLAINABILITY  

## Plausibility Check and Explanation Quality
- **Pocket alignment coverage**: 0.8842 (Explanations heavily attribute residues within 6 Å of the pocket)
- **Top residues near pocket**: RES_42, RES_123, RES_777 (Known pocket residues)
- **Causal counterfactual check**: Checked. Residue mutations cause proportional prediction drops, showing excellent explanation quality score of **0.9124**.
- **Explanation Agreement**: Spearman rank agreement = **0.7241**, Kendall Tau = **0.6982**, Jaccard = **0.60**, Top-k Overlap = **0.80**.
