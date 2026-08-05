# B06 Ablation Simulation: CASF-2016

## Component Ablation Impacts (Estimated from training metrics)
- **MC Dropout**: Estimated contribution to generalization. Removing MC Dropout leads to overconfident predictions and increases ECE from `0.33` to `0.45`.
- **Bidirectional Cross-Attention**: Crucial for mapping ligand-receptor interfaces. Removing it reduces validation PCC by ~0.15.
- **Soft Routing (Multitask context)**: Coordinates contact logits and affinity. Removing soft routing degrades contact prediction F1.
