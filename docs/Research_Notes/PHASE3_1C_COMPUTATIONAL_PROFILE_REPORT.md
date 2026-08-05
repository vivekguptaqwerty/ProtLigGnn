# Phase 3.1C Computational Profile Report
**Experiment ID**: `P3.1C-INTERACTION-REPRESENTATION`  
**Date**: 2026-07-16  

This report compares parameter overhead and memory usage across versions.

---

## 1. Parametric Comparison

- **Baseline v1.1**: 1,413,889 parameters
- **Attention v1.3**: 1,414,022 parameters
- **Interaction v1.4**: 1,455,974 parameters
- **IRM Candidate Overhead**: **41,952 parameters** (+2.97% overhead)
- **Peak GPU VRAM**: 985.0 MB (v1.3: 920.0 MB)
- **Inference Latency per sample**: 2.1 ms (v1.3: 1.8 ms)