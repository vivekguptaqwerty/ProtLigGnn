# Rollback Report: LigProtGNN-X v4.0

## Rollback Audit Summary
- **Target Version**: LigProtGNN-X v4.0
- **Action Taken**: Safe architectural rollback of v5 experimental concepts.
- **Files Archived to `docs/archive/`**:
  - `E01_v5_Research_Redesign.md`
  - `LigProtGNN_v5_Architecture_Blueprint.md`
  - `LigProtGNN_v5_1_Ultimate_Blueprint.md`
  - `LigProtGNN_v5_2_Ultimate_Blueprint.md`
  - `LigProtGNN_v5_3_Ultimate_Blueprint.md`
- **Files Modified**:
  - `config/config.yaml` (removed `adaptive_radius` parameters and updated experiment names)
- **Infrastructure Preserved**:
  - Core datasets registry and thread-safe loading infrastructure.
  - Immutable PDB/mmCIF structural data structures and validation APIs.
