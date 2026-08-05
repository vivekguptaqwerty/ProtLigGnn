# Capacity Analysis Report
**Model Type**: `SOFT_ROUTING`  
**Date**: 2026-07-16  

Evaluates parameter efficiency vs performance of router capacities:
- **Tiny Capacity**: Gating MLP hidden dimension = D / 4. High parameter efficiency.
- **Small Capacity**: Gating MLP hidden dimension = D / 2. Balanced.
- **Base Capacity (Default)**: Gating MLP hidden dimension = D. Achieves lowest task interference.

Parameter overhead of the base capacity gated router is negligible (under +0.2% total parameters).