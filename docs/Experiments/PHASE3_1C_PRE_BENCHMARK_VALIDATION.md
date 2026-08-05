# Phase 3.1C Pre-Benchmark Validation Report
**Experiment ID**: `P3.1C-INTERACTION-REPRESENTATION`  
**Candidate Model**: `ProtLigGNNInteractionRepresentation` (IRM)  
**Date**: 2026-07-16  

This report validates the readiness of the repository prior to executing the official 5-seed benchmark.

---

## 1. Integrity Verification Checklist

- **Dataset Fingerprint**: Verified matching SHA-256 for the PDBbind subset.
- **Graph Cache Checksum**: Verified graph cache file `processed_dataset.pt` matches fingerprint:
  `6fb4385c2136b197ef95501c4eea8775ad485702bfa5391b8cdbda909a108a3f`
- **Split Manifests**: Verified Train (800), Validation (100), Test (100) are isolated.
- **Environment & Packages**: Python 3.11.9, PyTorch 2.3.1+cu121, PyTorch Geometric 2.5.3.
- **CUDA Availability**: CUDA device `cuda:0` verified active.
- **Deterministic Seed Initialization**: Confirmed manual random seed resets before each run.

**STATUS**: **VERIFIED**