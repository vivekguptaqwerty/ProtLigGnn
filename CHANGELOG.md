# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-18
### Added
- central master configuration under `config/config.yaml`.
- centralized log handlers setup under `ligprotgnnx/utils/logger.py`.
- initialization unit tests under `tests/test_phase0.py`.
- `IMPLEMENTATION_LOG.md` and `CHANGELOG.md` for project progress tracking.
- `ligprotgnnx/datasets/registry.py` and `tests/test_dataset_registry.py` for Dataset Registry implementation.
- `ligprotgnnx/preprocessing/protein_structure.py` and `tests/test_protein_structure.py` for immutable ProteinStructure data model.
- `ligprotgnnx/preprocessing/{exceptions.py,file_detector.py,parser_registry.py,protein_parser.py}` and corresponding tests for Universal Protein Parser implementation.
### Changed
- Reverted repository state and Hydra configurations to represent the frozen LigProtGNN-X v4.0 architecture.
- Archived all v5 experimental research specs under `docs/archive/`.
