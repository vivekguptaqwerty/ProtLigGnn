# A07 Regression Head Analysis: CASF-2016

## Regressor MLP Weights & Biases
- **Layer 6 weight**: mean=`0.000187`, std=`0.04975`
- **Layer 6 bias**: `[-0.03610]`
- **Layer 3 weight**: mean=`0.000132`, std=`0.02493`
- **Layer 0 weight**: mean=`-0.000048`, std=`0.01781`

## Bias vs Scale Analysis
The regression head layers contain standard deviations and biases that are perfectly normal. However, because the model was trained for only 2 epochs on 10 samples (a tiny smoke test), it has converged to output a constant value close to the bias of the first epoch's average output.
