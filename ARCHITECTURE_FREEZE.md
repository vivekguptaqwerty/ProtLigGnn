# Architecture Freeze: LigProtGNN-X v4.0

## Status: FROZEN
- **Designated Version**: LigProtGNN-X v4.0
- **Status Date**: 2026-07-18
- **Architectural Constraints**:
  - No dynamic graph rewiring.
  - No adaptive radii or dynamic neighborhood calculations.
  - No edge memory networks or gated edge updates.
  - No hybrid physics learned residuals.
  - All experimental v5.x concept specifications are officially archived.

All future development, preprocessing, model forward loops, training, and evaluation scripts must strictly adhere to the frozen LigProtGNN-X v4.0 specification.
