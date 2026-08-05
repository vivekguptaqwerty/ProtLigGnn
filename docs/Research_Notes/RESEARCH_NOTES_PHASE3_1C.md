# Research Notes: Phase 3.1C — Edge-Aware Interaction Attention
**Experiment ID**: `P3.1C-EDGE-AWARE-INTERACTION`  

---

## 1. Scientific Background
Distance-biased attention (Phase 3.1B) assumes that the spatial proximity between a ligand atom and a protein residue center is the primary regulator of binding strength. However, molecular interface binding is driven by chemical compatibility (e.g. hydrogen bonds, salt bridges, van der Waals forces, and electrostatic interactions). 

By upgrading the model to learn a joint representation from geometry, chemistry, and representation branches, we allow the attention logits to be regularized by specific chemical combinations (e.g., placing an aromatic donor near an aromatic acceptor increases logit alignment more than placing a hydrophobic carbon near a polar oxygen at the same physical distance).

---

## 2. Hypotheses Mapping
- **H0**: Adding chemical/representational features yields no statistical prediction improvements over Baseline v1.3.
- **H1**: Learnable interaction-aware attention reduces test RMSE.
- **H2**: Learnable interaction-aware attention improves Ranking/PCC metrics.
- **H3**: Chemistry-aware attention bias resolves spatial attention localization, aligning attention weights with physically meaningful contacts.
