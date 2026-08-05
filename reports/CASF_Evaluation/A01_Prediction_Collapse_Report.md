# A01 Prediction Collapse Report: CASF-2016

## Diagnosis Metrics (N = 285 complexes)
- **Experimental Target Mean**: 6.4864
- **Experimental Target Std**: 2.1705
- **Experimental Target Variance**: 4.710931
- **Experimental Target Range**: [2.0700, 11.8200] (Dynamic Range = 9.7500)
- **Experimental Target Coefficient of Variation**: 0.3346
- **Experimental Target Entropy (Gaussian Nats)**: 2.1939

- **Model Prediction Mean**: 3.8358
- **Model Prediction Std**: 0.0744
- **Model Prediction Variance**: 0.005543
- **Model Prediction Range**: [3.5661, 4.0564] (Dynamic Range = 0.4903)
- **Model Prediction Coefficient of Variation**: 0.0194
- **Model Prediction Entropy (Gaussian Nats)**: -1.1787

## Statistical Comparison
- **Variance Reduction**: Predictions exhibit a **99.88% reduction in variance** compared to experimental targets.
- **Kolmogorov-Smirnov Test**:
  - KS Statistic: `8.5965e-01`
  - p-value: `8.0343e-109`
  - Verdict: **Extremely significant difference**. The hypothesis that targets and predictions originate from the same distribution is rejected.

## Collapse Target Analysis
The predictions are collapsed to a narrow range around **3.836**, which matches the mean prediction of the smoke test training run checkpoint `run_001` (validation mean prediction: `3.8609`) trained on `max_samples: 10`.
