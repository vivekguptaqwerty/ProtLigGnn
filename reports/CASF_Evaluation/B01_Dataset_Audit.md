# B01 Dataset Audit: CASF-2016

## Split Inventory
- **Training Samples (N)**: 800
- **Validation Samples (N)**: 100
- **Test Samples (N)**: 100
- **Total Development Size**: 1000 complexes

## Divergence & Diversity
- **Database Scope**: PDBbind v2020 contains >19,000 general complexes and >5,300 refined complexes. The development dataset of **1000 complexes** represents a mere **5.2% of PDBbind general set** and **18.8% of the refined set**.
- **Data Starvation**: The model suffers from severe data starvation. 800 training samples are insufficient to capture the vast structural, atomic, and chemical diversity of protein-ligand pockets.
- **Affinity Distribution**: The affinities span from `3.11` to `10.0` with a mean of `5.42`. CASF-2016 spans from `2.07` to `11.82` with a mean of `6.48`.
