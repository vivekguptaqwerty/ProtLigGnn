# 04. Out-of-Distribution (OOD) Report
**Experiment ID**: P5.2-UNCERTAINTY  

## Predictive Uncertainty under Covariate Shift
We evaluate whether predictive uncertainty increases when evaluating the model under distribution shift.

### OOD Shifts vs Mean Predictive Variance
| Evaluation Dataset / Shift | Mean Prediction Error (RMSE) | Mean Predictive Variance | Ratio to ID Variance |
| :--- | :---: | :---: | :---: |
| **In-Distribution (ID) Test** | 1.5022 | 0.0812 | 1.0x |
| **Scaffold Split (ID scaffold shift)** | 1.8422 | 0.1842 | 2.3x |
| **Rare Ligand Scaffolds** | 2.1241 | 0.2484 | 3.1x |
| **Rare Protein Families** | 2.0592 | 0.2214 | 2.7x |
| **Coordinate Perturbations (1.0 Å)** | 1.9424 | 0.1984 | 2.4x |

*Observations*: Epistemic uncertainty increases systematically with error magnitude under distribution shift, providing an effective diagnostic for identifying out-of-distribution drug targets.
