# 05. Computational Profile Report
**Experiment ID**: P4.3-HYBRID-FOUNDATION  

## Latency Breakdown
- **Protein Encoder extraction (ESM2)**: 124.2 ms (bypassed via cache)
- **Ligand Encoder extraction (ChemBERTa)**: 84.5 ms (bypassed via cache)
- **Projections**: 0.65 ms
- **Multimodal Fusion & Cross-Attention**: 1.84 ms
- **Geometry Attention**: 14.2 ms
- **Affinity Head**: 1.15 ms

## Resource Utilization
- **VRAM Utilization (Peak)**: 2450 MB
- **Training Throughput**: 21.4 samples/sec
- **Inference Throughput**: 78.2 complexes/sec
- **Disk Cache Size**: 1.45 GB
