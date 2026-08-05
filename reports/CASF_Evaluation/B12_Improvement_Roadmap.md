# B12 Improvement Roadmap: CASF-2016

## Recommendations (Without Changing Architecture)
1. **Expand Dataset to Full PDBbind Refined Set (N = 5,316)**
   - **Expected Effect**: Pearson R: ~0.50 - 0.55, RMSE: ~1.6 - 1.8.
   - **Engineering Effort**: Medium (regenerating graph cache).
2. **Implement Cosine Annealing LR Scheduler & Warmup**
   - **Expected Effect**: Reduces overfitting, smoother loss convergence.
3. **Optimize Distance Cutoff for Pocket Extraction**
   - **Expected Effect**: Reducing cutoff from 6.0Å to 4.5Å focuses representation on contact-relevant residues.
