# 02. Interface Report
**Experiment ID**: P5.5-INTEGRATION  

## Unified API Specifications
The unified inference API exposing `model.predict()` has been validated. All calls return a structured `PredictionResult` carrying:
- **affinity**: Regressed target affinity.
- **confidence**: Uncertainty confidence score.
- **prediction_interval**: Lower/upper confidence interval bounds.
- **uncertainty**: Standard deviation parameter.
- **explanation**: ExplanationResult object with residue and atom rankings.
- **robustness**: Composite Robustness Index (CRI) score.
- **ood_probability**: Out-of-distribution probability classification.
- **metadata**: Log/method version.
