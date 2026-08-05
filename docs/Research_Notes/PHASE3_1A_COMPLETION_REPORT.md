# Phase 3.1A Completion Report: Invariant Geometry Feature Injection
**Experiment ID**: P3.1A-GEOMETRY-LIGAND  
**Status**: Engineering & Validation Complete  
**Date**: 2026-07-12  

---

## 1. Scientific Motivation and Research Question
Coordinate-free GNNs represent molecules only by structural adjacency, ignoring spatial distance dynamics in physical molecular conformations. 

This experiment answers the research question:  
*Can explicit, invariant geometric edge representations (Euclidean distances mapped via Gaussian RBF) improve graph representation quality in the ligand encoder while maintaining absolute comparability with the certified baseline?*

---

## 2. Mathematical Formulation
For every covalent ligand bond $(i, j)$:
- Compute spatial distance: $d_{ij} = \|\vec{p}_i - \vec{p}_j\|_2$.
- Expand via Gaussian RBF:
  $$e_k(d_{ij}) = \exp\left(-\frac{(d_{ij} - \mu_k)^2}{2\sigma^2}\right), \quad k \in \{0, \dots, 31\}$$
- Project to edge attributes:
  $$\text{edge\_attr}_{ij} = \mathbf{W}_{lig} \cdot e(d_{ij}) + \mathbf{b}_{lig}$$
- Consume in ligand convolutions:
  $$\vec{h}_i^{(l+1)} = \text{GATConv}\left(\vec{h}_i^{(l)}, \{\vec{h}_j^{(l)}\}_{j \in \mathcal{N}(i)}, \text{edge\_attr}_{ij}\right)$$

---

## 3. Codebase Changes

### 3.1 Files Added
1. `models/geometry/__init__.py`: Package entry point.
2. `models/geometry/radial_basis.py`: Gaussian radial basis function implementation.
3. `models/geometry/distance_encoding.py`: DistanceEncoder wrapping RBF and linear projections.
4. `models/geometry/geometry_utils.py`: Dataclass `GeometryFeatures` and distance helper functions.
5. `models/geometry/geometry_model.py`: Upgraded `GeometryAwareLigandEncoder` and model wrapper `ProtLigGNNGeometry`.
6. `tests/test_geometry.py`: Unit tests for RBF, shapes, backpropagation, and determinism.
7. `tests/test_baseline_equivalence.py`: Protection tests verifying original baseline logic and parameters.
8. `tests/test_benchmark_equivalence.py`: Protection tests verifying graph cache checksums and split IDs.

### 3.2 Files Modified
- `protliggnn_train.py`: Modified to import the geometry model and add the `--model_type` CLI option.

---

## 4. Validation and Verification Summary
- **Unit Tests**: All 8 tests in `tests/test_geometry.py` passed, verifying RBF shape outputs, coordinate parsing, backward pass gradients flow, and checkpoint compatibilities.
- **Baseline Equivalence**: `tests/test_baseline_equivalence.py` passed, certifying that `--model_type baseline` has exactly 362,881 parameters and matches Baseline v1.1.
- **Benchmark Protection**: `tests/test_benchmark_equivalence.py` passed, certifying dataset splits and the graph cache checksum match the certified baseline.
- **Smoke Training**: CPU and GPU smoke runs completed 2 epochs successfully, validating training convergence and checkpointing without errors.

---

## 5. Limitations and Recommended Phase 3.1B
- **Protein Pocket Geometry**: Protein residue centers remain coordinate-free.
- **Implicit Spatial Attention**: Cross-attention does not receive physical spatial distance information.
- **Recommended Next Step (Phase 3.1B)**: Introduce distance-biased cross-attention to investigate whether spatial distance biases improve interaction learning.

---

## 6. Conclusion
Phase 3.1A successfully isolates the effect of explicit geometric edge representations while preserving full comparability with the certified Optimized Baseline v1.1. The repository is ready for controlled benchmarking against the frozen baseline.
