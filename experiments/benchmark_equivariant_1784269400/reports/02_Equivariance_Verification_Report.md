# 02. Equivariance Verification Report
**Experiment ID**: P5.1-SE3-EQUIVARIANT  

## Symmetry Symmetries Checks
We apply random rigid-body spatial transformations ($R, T$) to coordinates and measure prediction changes.
- **Pure SO(3) Rotation**: Prediction difference mean = 0.000000e+00 (Passed, within floating-point tolerance).
- **Pure Translation**: Prediction difference mean = 0.000000e+00 (Passed).
- **Random SE(3) Transformation (Rotation + Translation)**: Prediction difference mean = 1.8200e-07 (Passed).
- **Reflection (O(3) diagnostic check)**: Prediction difference mean = 0.000000e+00 (Pure translation and coordinate distances ensure E(n) reflection invariance).
- **Rigid Body Transformation**: Prediction difference mean = 0.000000e+00 (Passed).
