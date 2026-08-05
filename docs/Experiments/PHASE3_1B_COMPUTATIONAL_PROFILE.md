# Phase 3.1B Computational Profile Report
**Experiment ID**: `P3.1B-GEOMETRY-ATTENTION`  
**Status**: Evaluation Completed  
**Date**: 2026-07-15  

This report evaluates the computational, memory, and checkpoint size overhead of the distance-biased cross-attention layer compared to the Baseline v1.1 and Phase 3.1A models.

---

## 1. Computational Profile Matrix

The profile was executed on an NVIDIA GeForce RTX 3050 Laptop GPU using a batch size of 8.

| Performance Metric | Certified Baseline v1.1 | Geometry v1.2 (P3.1A) | Geometry Attention v1.3 (P3.1B) | P3.1B vs. Baseline Delta (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Total Parameters** | 362,881 | 376,609 | 363,014 | **+0.037%** |
| **Trainable Parameters** | 362,881 | 376,609 | 363,014 | **+0.037%** |
| **State Dict Disk Size** | 1.4011 MB | 1.4566 MB | 1.4031 MB | **+0.143%** |
| **Inference Latency (Batch of 8)** | 17.35 ms | 19.00 ms | 25.91 ms | **+49.34%** |
| **Mean Latency per Sample** | 2.17 ms | 2.38 ms | 3.24 ms | **+49.34%** |
| **Peak GPU VRAM (MB)** | 24.14 MB | 26.94 MB | 31.49 MB | **+30.45%** |

---

## 2. Overhead Interpretation

### 2.1 Parameters & Space Efficiency
Phase 3.1B is highly parameter-efficient. By keeping the encoders coordinate-free and injecting the spatial distances directly inside the cross-attention interaction block, we only introduce **133 extra parameters** (vs 13,728 extra parameters in Phase 3.1A). The disk checkpoint size increases by only 2 KB.

### 2.2 Inference and VRAM Cost
We observe an inference latency increase of **+8.56 ms** (+49.3%) and a peak VRAM increase of **+7.35 MB** (+30.4%). This overhead stems from:
1. Dynamic pairwise Euclidean distance computations:
   $$\mathbf{D}_{ij} = \|\vec{p}_{lig, i} - \vec{p}_{prot, j}\|_2$$
2. Computing the RBF expansion and projected linear weight masks in python at runtime.
3. Broadcasting and applying the `attn_mask` matrix inside PyTorch's `nn.MultiheadAttention` module.

The computational overhead is extremely small in absolute terms, ensuring that the model runs fast during both training and high-throughput inference scenarios.
