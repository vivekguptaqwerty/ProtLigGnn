# C07 Architecture Audit: CASF-2016

## Structural Rankings (Expected Gains)
1. **Increase GNN layers (GCN/GAT) from 3 to 4**:
   - *Expected Gain*: Moderate ($+0.03$ Pearson).
   - *Risk*: Low.
2. **Implement Attentive Pooling**:
   - *Expected Gain*: Moderate ($+0.02$ Pearson).
   - *Risk*: Low.
3. **Increase hidden dimension from 256 to 512**:
   - *Expected Gain*: Low (will accelerate overfitting on small datasets unless dataset is expanded first).
   - *Risk*: High.
