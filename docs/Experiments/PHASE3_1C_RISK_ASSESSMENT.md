# Phase 3.1C Risk Assessment & Mitigation
**Experiment ID**: `P3.1C-EDGE-AWARE-INTERACTION`  
**Status**: Proposed  

This document evaluates the software engineering and scientific risks associated with the Interaction Representation Module.

---

## 1. Identified Risks

### 1.1 Overfitting Risk (High Scientific Risk)
- **Description**: The IRM introduces more parameters and capacity than Baseline v1.3. On small datasets (1,000 complexes), this extra capacity can lead to overfitting on the training splits, increasing test set RMSE.
- **Mitigation**: Keep branch dimensions low (default `32`), apply dropout ($0.1$) inside the fusion MLP, and lock GAT/GCN encoder learning rates.

### 1.2 Computational Latency Risk (Medium Engineering Risk)
- **Description**: Constructing and forwarding $L \times P \times F$ tensors through multi-branch MLPs at every epoch increases training time.
- **Mitigation**: Implement optional caching (`cache_interactions=True`). For static graphs, raw features and distances do not change across epochs, allowing us to cache and reuse the raw branch representations.

### 1.3 Numerical Instability Risk (Medium Scientific Risk)
- **Description**: Logits with large magnitudes can cause exponent overflow or vanishing gradients during Softmax computation.
- **Mitigation**: Apply LayerNorm to project output biases and scale them using learned log-space parameters $\alpha$.
