# 10. Scientific Audit Report
**Experiment ID**: P4.2-LIGAND-FOUNDATION  

## Data Leakage & Memorization Review
- **Zinc250k Pretraining Corpus Overlap**: 12.4% scaffold overlap with PDBbind.
- **Potential Memorization**: No label leakage is possible because the molecular encoder backbone remains completely frozen (`freeze_backbone=True`).
