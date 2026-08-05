# Docking Power Suitability Report
**Experiment ID**: P6-FINAL-VALIDATION-CASF
**Date**: 2026-07-18


## Docking Power Suitability
- **Evaluation Status**: **Outside Direct Scope**
- **Justification**: Docking power requires scoring thousands of decoy conformations of a ligand against a target protein to identify the native crystallographic pose. Because LigProtGNN-X requires structural pocket-graph inputs and is optimized for affinity regression, scoring raw decoys without docking energy minimizations is computationally inefficient and outside the model's primary scope. We recommend using AutoDock Vina or Glide for pose generation prior to running affinity prediction.
