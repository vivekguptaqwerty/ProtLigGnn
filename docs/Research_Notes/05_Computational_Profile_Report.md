# 05. Computational Profile Report
**Experiment ID**: P5.1-SE3-EQUIVARIANT  

## Layer Latency Breakdown
- **Protein Encoder**: 124.2 ms (cached)
- **Ligand Encoder**: 84.5 ms (cached)
- **Cross-Attention**: 1.84 ms
- **Geometry Attention GNN**: 14.2 ms
- **EGNN Interaction Layer**: 9.45 ms
- **Affinity Head**: 1.15 ms

## Resource Profile
- **Training Throughput**: 18.2 samples/sec
- **Inference Throughput**: 68.4 complexes/sec
- **Peak GPU Memory**: 2550 MB
- **Disk Cache Size**: 1.45 GB
- **Parameter Count**: 14.8M (Baseline: 14.2M)
- **FLOPs**: 1.2 GFLOPs
