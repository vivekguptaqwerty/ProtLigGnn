# Phase 3.1B Pre-Benchmark Validation Report
**Experiment ID**: `P3.1B-GEOMETRY-ATTENTION`  
**Certified Date**: 2026-07-15  
**Validation Status**: **PASSED**  

This document certifies that the environment, inputs, and validation splits are correctly configured prior to starting the official 5-seed benchmark.

---

## 1. Validation Checklist & Status

1. **Graph Cache Integrity**:
   - Cache file: `data/pdbbind2020/processed_dataset.pt`
   - Expected SHA-256: `6fb4385c2136b197ef95501c4eea8775ad485702bfa5391b8cdbda909a108a3f`
   - Computed SHA-256: `6fb4385c2136b197ef95501c4eea8775ad485702bfa5391b8cdbda909a108a3f`
   - **Status**: **PASS**

2. **Dataset Manifest**:
   - Verified that `data/pdbbind2020/graph_cache_manifest.json` exists and is formatted correctly.
   - **Status**: **PASS**

3. **Benchmark Seeds**:
   - Seeds matched exactly: `[42, 123, 777, 2024, 3407]`
   - **Status**: **PASS**

4. **Split Manifests**:
   - `splits/train_ids.txt` (800 lines), `splits/val_ids.txt` (100 lines), `splits/test_ids.txt` (100 lines) exist and are non-empty.
   - **Status**: **PASS**

5. **Registry Integrity**:
   - Checked that `experiments/registry.csv` is present and consistent.
   - **Status**: **PASS**

6. **Environment & CUDA Availability**:
   - Python: `3.11.9`
   - PyTorch: `2.3.1+cu121`
   - PyTorch Geometric: `2.5.3`
   - CUDA GPU: Available (`NVIDIA GeForce RTX 3050 6GB Laptop GPU`).
   - **Status**: **PASS**

7. **Checkpoint Compatibility & Configuration Serialization**:
   - Checked checkpoint serialization via `test_geometry_attention_checkpoint.py`. Configuration dataclass parses to and from dicts correctly.
   - **Status**: **PASS**

---

## 2. Conclusion

Based on the 100% passing rate of the pre-validation integrity audit:

### **STATUS: AUTHORIZED TO BENCHMARK**
