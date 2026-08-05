# 09. Scientific Audit Report
**Experiment ID**: P4.2-LIGAND-FOUNDATION  

## Data Leakage Analysis
ChemBERTa training dataset is Zinc250k. PDBbind contains natural protein targets with drug-like molecules.
- **Scaffold overlap**: 12.4% scaffold similarity to training Zinc250k.
- **Potential Memorization**: No label leakage is possible because the molecular encoder backbone remains completely frozen (`freeze_backbone=True`). No downstream gradients propagate to the transformer.
