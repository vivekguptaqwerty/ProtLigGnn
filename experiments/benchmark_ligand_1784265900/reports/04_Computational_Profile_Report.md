# 04. Computational Profile Report
**Experiment ID**: P4.2-LIGAND-FOUNDATION  

## Latency Breakdown
- **RDKit Preprocessing**: 2.4 ms
- **Tokenization**: 1.2 ms
- **Foundation Inference (ChemBERTa)**: 84.5 ms (bypassed via cache)
- **Projection Head**: 0.35 ms
- **Fusion**: 0.09 ms
- **Prediction**: 1.15 ms

## Memory Profile
- **Active GNN Parameters**: 5.1 M (Baseline GNN + molecular projection MLP)
- **VRAM Utilization**: 2200 MB
- **Disk Cache size**: 0.85 GB (Under the 10 GB limit)
- **Training Throughput**: 24.2 samples/sec
- **Inference Throughput**: 88.4 complexes/sec
