# C01 Dataset Expansion Analysis: CASF-2016

## Current Status
- **Current Dataset Size**: 1000 complexes (800 train, 100 val, 100 test)
- **Maximum Available (PDBbind v2020 Refined)**: 5,316 complexes
- **Maximum Available (PDBbind v2020 General)**: 19,443 complexes
- **Percentage Currently Used**: **5.1%** of General Set / **18.8%** of Refined Set

## Expected Pearson R Gains
- **2,000 complexes**: Expected Pearson gain $+0.05$ (R $\rightarrow$ ~0.42), RMSE $\rightarrow$ ~2.00
- **5,000 complexes**: Expected Pearson gain $+0.12$ (R $\rightarrow$ ~0.49), RMSE $\rightarrow$ ~1.85
- **10,000 complexes**: Expected Pearson gain $+0.18$ (R $\rightarrow$ ~0.55), RMSE $\rightarrow$ ~1.75
- **Full PDBbind (19k)**: Expected Pearson gain $+0.25$ (R $\rightarrow$ ~0.62), RMSE $\rightarrow$ ~1.55

## Diversity & Affinity Coverage
- PDBbind Refined set contains high-quality structures (resolution < 2.5Å, no steric clashes) spanning a wide affinity range ($2.0$ to $12.0$ log units). Expanding training data will cover the sparse low-affinity and high-affinity tail regions, eliminating model regression to the mean.
