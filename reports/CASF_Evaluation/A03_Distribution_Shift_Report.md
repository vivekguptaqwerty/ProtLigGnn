# A03 Distribution Shift Analysis: CASF-2016

## Distribution Comparison
- **Training Set Cache (N = 50)**:
  - Mean: 5.4256
  - Median: 5.3768
  - Std: 1.4070
  - Min: 3.1118
  - Max: 10.0000
- **CASF-2016 Core Set (N = 285)**:
  - Mean: 6.4864
  - Median: 6.4800
  - Std: 2.1705
  - Min: 2.0700
  - Max: 11.8200

## Statistical Distance (PDBbind subset vs CASF-2016)
- **Kolmogorov-Smirnov Test**:
  - KS statistic: `0.2912`
  - p-value: `0.0011`
- **Verdict**: A statistically significant distribution shift exists between the PDBbind training cache subset (max_samples=50) and the CASF-2016 benchmark. However, this shift cannot explain the **prediction collapse** because the model predictions also collapse on the training set itself (Pred Std: `0.0372` on training set).
