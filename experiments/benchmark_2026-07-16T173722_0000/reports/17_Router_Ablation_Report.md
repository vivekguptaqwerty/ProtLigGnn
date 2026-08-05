# Router Ablation Report
**Model Type**: `SOFT_ROUTING`  
**Date**: 2026-07-16  

Comparative evaluation of task router modules:
1. **Gated Router (Default)**: Outperforms linear and residual projections in training stability and RMSE.
2. **Linear Router**: High task-divergence but limits joint feature regularization.
3. **Residual Router**: Preserves shared dimensions but shows slightly higher gradient conflict.

Gated routing successfully balances representation preservation and gradient mitigation.