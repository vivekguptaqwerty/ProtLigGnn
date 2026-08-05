# A04 Feature Pipeline Audit: CASF-2016

## Pipeline Consistency Checklist
- [x] Ligand representation (SDF -> MOL2 fallback logic verified matching training features)
- [x] Protein pocket definition (distance cutoffs at 6.0Å)
- [x] Graph node features (ligand = 78 dimensions, protein = 30 dimensions)
- [x] Graph edge features (matching coordinate distance calculations)
- [x] Embeddings usage (ESM-2 / ChemBERTa features are not utilized by `outputs/best_protliggnn.pt` checkpoint)

## Mismatch Audit
- No preprocessing feature mismatch detected. The graph features and feature ordering during evaluation are identical to the training dataset preprocessing.
