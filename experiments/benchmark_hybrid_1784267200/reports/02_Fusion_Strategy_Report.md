# 02. Fusion Strategy Report
**Experiment ID**: P4.3-HYBRID-FOUNDATION  

## Alignment & Fusion Quality
We analyze the stability and capacity of the five implemented fusion strategies:
- **Bidirectional Cross-Attention**: Exhibits stable loss trajectories, zero convergence failures, and the highest validation performance. Effectively aligns localized pocket representations with molecular subgraphs.
- **Gated Fusion**: Learns to gate protein vs ligand features dynamically, showing higher alignment accuracy than concat/weighted sum, but marginally inferior to cross-attention.
- **No-Fusion Control**: Late ensemble of unimodal prediction heads converges but is unable to capture explicit molecular contacts, leading to sub-optimal affinity resolution.
