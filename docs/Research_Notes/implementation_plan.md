# Implementation Plan — Phase 5.6: Repository Organization & Documentation

This document outlines the organization and centralization plan to structure the `LigProtGNN-X` workspace, moving experimental scripts, clean documentation, and reports into standard directories.

## 1. Objectives

- **Centralize Documentation**: Move and consolidate all scattered markdown files into the `docs/` folder, structured logically.
- **Establish registries**:
  - **Experiment Registry**: Standardize folders in `experiments/` for Phase 4.1 to Phase 5.5, each with a config, results, and quick summary.
  - **Dataset Registry**: Create `data/metadata/` registry files (`dataset_registry.yaml`, `dataset_description.md`, etc.).
- **Archive Experimental Code**: Move legacy models, EGNN tests, and prototype scripts to `archive/`.
- **Root Cleanup**: Remove obsolete reports, logs, and transient scripts, keeping only `README.md`, `ROADMAP.md`, `CHANGELOG.md`, `PROJECT_STRUCTURE.md`, and `requirements.txt` at the root.

---

## 2. Target Directory Structure

The repository will be structured as follows:

```
LigProtGNN-X/
├── README.md
├── ROADMAP.md
├── CHANGELOG.md
├── PROJECT_STRUCTURE.md
├── requirements.txt
│
├── src/
│   └── ligprotgnnx/
│       ├── core/
│       ├── inference/
│       ├── configs/
│       ├── utils/
│       └── __init__.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── cache/
│   ├── metadata/
│   │   ├── dataset_registry.yaml
│   │   ├── dataset_description.md
│   │   └── dataset_statistics.md
│   └── README.md
│
├── experiments/
│   ├── Phase_4.1/
│   ├── Phase_4.2/
│   ├── Phase_4.3/
│   ├── Phase_5.1/
│   ├── Phase_5.2/
│   ├── Phase_5.3/
│   ├── Phase_5.4/
│   └── Phase_5.5/
│
├── reports/
│   ├── Phase_4/
│   └── Phase_5/
│
├── docs/
│   ├── Architecture.md
│   ├── Datasets.md
│   ├── Benchmarking.md
│   ├── Methodology.md
│   ├── Model_Cards.md
│   ├── Reliability.md
│   ├── Explainability.md
│   ├── Robustness.md
│   └── Project_Structure.md
│
└── archive/
    └── Phase_5.1_EGNN/
```

---

## 3. Proposed Changes

We will execute the folder creation and move commands, standardizing naming and deleting obsolete files.

### registries & configs
- Create `data/metadata/dataset_registry.yaml` mapping PDBbind structures and stats.
- Create `docs/Architecture.md` describing geometry attention GNN.
- Create `docs/Model_Cards.md` describing v4.0 baseline model specifications.

### Code Cleanup
- Scan and delete stale logs (`*.log`), temp notebooks, and deprecated print statements.
- Verify standard imports inside the source package.

---

## 4. Verification Plan

### Automated Checks
- Run pytest verification suite:
  `.\venv\Scripts\python -m pytest tests/ -v`

### Manual Verification
- Verify directories layout matches requirements.
- Verify root-level cleanliness.
