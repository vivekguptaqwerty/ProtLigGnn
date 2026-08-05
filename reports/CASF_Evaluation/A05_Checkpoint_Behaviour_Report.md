# A05 Checkpoint Behaviour: CASF-2016

## Multi-Split Inference Comparison
1. **Training Split (10 random samples)**:
   - Prediction Mean: 3.841
   - Prediction Std: 0.032
   - Target Std: 1.254
2. **Validation Split (10 samples)**:
   - Prediction Mean: 3.838
   - Prediction Std: 0.035
   - Target Std: 1.311
3. **CASF Split (10 samples)**:
   - Prediction Mean: 3.834
   - Prediction Std: 0.041
   - Target Std: 2.215

## Localization of Collapse
The collapse does **not** begin during CASF evaluation. The collapse exists uniformly across training, validation, and benchmark splits. This confirms the model weights themselves represent a collapsed constant predictor.
