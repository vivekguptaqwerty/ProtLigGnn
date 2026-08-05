# B03 Prediction Analysis: CASF-2016

## Distribution Comparison
| Dataset | Count | Mean Prediction | Std Prediction | Pearson R | RMSE | MAE | R² |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Validation/Test (PDBbind)** | 100 | 5.617 | 0.803 | 0.417 | 1.648 | 1.302 | 0.108 |
| **CASF-2016 Core Set** | 285 | 5.860 | 0.786 | 0.371 | 2.111 | 1.705 | 0.054 |

## Analysis
The model's predictions on the CASF benchmark are not collapsed (Std: **0.786** vs targets Std: `2.17`), but the overall correlation is modest (Pearson R = **0.371**). This is primarily because the model was trained on a small subset (800 samples) and suffered from severe validation divergence.
