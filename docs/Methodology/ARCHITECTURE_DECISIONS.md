# ProtLigGNN Architecture Decision Records

Date created: 2026-07-11

This file records why major modeling and research-infrastructure decisions exist. `BRAIN.md` remains the living project knowledge base; this file preserves decision rationale over time.

## ADR-001: Use Script-Compatible Infrastructure Modules Instead Of A Package Refactor

Decision:
- Add reusable modules such as `benchmarking.py` and `experiment_infra.py` while preserving the current script-based repository layout.

Context:
- The project is a research prototype with existing entrypoints such as `protliggnn_train.py`, `benchmark.py`, and `infer_affinity.py`.
- Backward compatibility is required for existing training, inference, and report scripts.

Alternatives considered:
- Refactor into an installable `protliggnn/` package.
- Keep adding all infrastructure directly inside `protliggnn_train.py`.

Why this option was chosen:
- It improves modularity without breaking imports or CLI workflows.
- It keeps the milestone focused on infrastructure rather than a broad repository migration.

Trade-offs:
- The repository still lacks package-level boundaries and tests.
- Some orchestration remains in `protliggnn_train.py`.

Future implications:
- A later package refactor should move these modules into a proper package with tests and typed interfaces.

## ADR-002: Generate Deterministic Split Manifests From Processed PDB IDs

Decision:
- Build train/validation/test manifests from complexes that successfully pass graph preprocessing.

Context:
- RDKit/BioPython graph construction can skip raw candidate complexes.
- Reproducibility requires the saved split to match the actual trainable dataset.

Alternatives considered:
- Split raw candidate records before preprocessing.
- Keep random in-memory index splitting.

Why this option was chosen:
- It prevents split manifests from containing samples that were never trainable.
- It makes reruns auditably reproducible.

Trade-offs:
- Split identity can change if preprocessing logic changes.
- Preprocessing failures must be tracked separately in `dataset_metadata.json`.

Future implications:
- Add preprocessing-version fingerprints so split provenance remains clear across graph-construction changes.

## ADR-003: Use Local Filesystem Experiment Directories And CSV Registry

Decision:
- Store each run under `experiments/run_XXX/` and maintain `experiments/registry.csv`.

Context:
- The project has no W&B, MLflow, database, or external tracking dependency.
- Publication reproducibility needs a local, inspectable artifact system.

Alternatives considered:
- Add MLflow or W&B.
- Store only per-run manifests without a global registry.

Why this option was chosen:
- It uses standard libraries and works offline.
- It is easy to inspect, version, archive, and compare.

Trade-offs:
- CSV does not handle concurrency or rich queries as well as a database.
- Manual cleanup can desynchronize the registry from deleted runs.

Future implications:
- If the project scales to many concurrent experiments, migrate the registry to SQLite while preserving CSV export.

## ADR-004: Hash Both Dataset Artifacts And Canonical Configurations

Decision:
- Write `dataset_hash.json`, `config_hash.txt`, and `config_fingerprint.json` for every tracked experiment.

Context:
- Previous retained metrics lacked enough provenance to detect dataset or configuration drift.
- Duplicate experiment detection requires stable identities.

Alternatives considered:
- Use run names as experiment identity.
- Hash only `config.json`.

Why this option was chosen:
- Dataset and config hashes capture what matters scientifically.
- Canonical config hashing ignores output-only identity fields so equivalent experiments get the same hash.

Trade-offs:
- Hash scopes must be maintained as new metadata fields appear.
- Historical runs without hashes cannot be deduplicated automatically.

Future implications:
- Add model-code and preprocessing-code hashes to detect implementation drift.

## ADR-005: Preserve Raw Hashes But Use Stable JSON Digests For Duplicate Detection

Decision:
- Record raw SHA256, file size, and modification time for auditability, and use stable JSON digests with volatile timestamp fields removed for overall dataset identity.

Context:
- Split and metadata JSON files contain timestamps.
- Raw JSON hashes would change across identical experiments due only to metadata timestamps.

Alternatives considered:
- Remove timestamps from all JSON artifacts.
- Use raw file hashes for duplicate detection.

Why this option was chosen:
- Raw forensic details remain available.
- Duplicate detection is not defeated by timestamp-only changes.

Trade-offs:
- Stable digest logic must be documented and maintained.
- A volatile-field list can miss future timestamp-like keys.

Future implications:
- Centralize provenance schemas so stable hashing rules are explicit and tested.

## ADR-006: Block Data Leakage Before Training

Decision:
- Validate split disjointness, duplicates, missing IDs, and orphan IDs before training begins.

Context:
- Data leakage invalidates benchmark claims.
- Earlier random splitting did not persist or validate split identity.

Alternatives considered:
- Warn but continue on leakage.
- Validate only after training.

Why this option was chosen:
- Scientific benchmark validity is more important than permissive execution.
- Failing early avoids wasted compute and invalid artifacts.

Trade-offs:
- Tiny smoke-test datasets with overlapping train/val/test IDs now fail tracked runs.
- Users must provide enough samples for strict splits.

Future implications:
- Add separate explicit smoke-test mode if tiny overlapping splits are needed for CI.

## ADR-007: Use Python Logging For Experiment Events

Decision:
- Configure timestamped Python logging and write `experiment.log` inside each experiment.

Context:
- Prior training used scattered `print` statements.
- Reproducible experiments need structured logs for CLI args, environment, training, validation, checkpoints, evaluation, and errors.

Alternatives considered:
- Continue writing ad hoc run logs.
- Add a third-party structured logging package.

Why this option was chosen:
- Standard-library logging is portable and dependency-free.
- It supports INFO, WARNING, ERROR, and DEBUG without changing runtime dependencies.

Trade-offs:
- Logs are text rather than fully structured JSON.
- Some third-party progress output may still appear on console.

Future implications:
- Add optional JSONL logging if automated log ingestion becomes necessary.

## ADR-008: Include Optimizer, Scheduler Slot, And RNG State In New Checkpoints

Decision:
- Save model state, optimizer state, optional scheduler state, split manifest, stats, and RNG state in new checkpoints.

Context:
- Reliable resume requires more than model weights.
- Existing inference expects `model_state_dict` and `args`.

Alternatives considered:
- Save only model weights.
- Introduce a separate resume checkpoint format.

Why this option was chosen:
- It preserves backward compatibility while enabling robust continuation.
- A scheduler slot exists even though no scheduler is currently used.

Trade-offs:
- Checkpoints become larger.
- Older checkpoints cannot restore optimizer/RNG state if they never stored it.

Future implications:
- Add tests that verify resumed training continuation in a Torch-enabled environment.

## ADR-009: Keep Experiment Comparison Artifact-Based

Decision:
- Implement `compare_runs.py` using saved run directories rather than live training objects.

Context:
- Comparisons must work on archived experiments and in environments without Torch.
- The current local machine cannot import Torch.

Alternatives considered:
- Compare by re-running evaluations.
- Compare through an external experiment tracking service.

Why this option was chosen:
- It is reproducible, offline, and simple to audit.
- It encourages complete artifact generation.

Trade-offs:
- It cannot compare missing artifacts.
- Single-run comparisons are descriptive, not statistically conclusive.

Future implications:
- Extend comparison to aggregate benchmark directories for stronger statistical analysis.

## ADR-010: Preserve The Existing ProtLigGNN Model Architecture

Decision:
- Do not redesign the GNN, add geometry-aware layers, change labels, or alter datasets in this infrastructure milestone.

Context:
- The milestone objective is reproducibility and benchmark robustness, not accuracy improvement.
- Retained results show the no-crossgraph ablation currently performs best, so modeling claims must remain conservative.

Alternatives considered:
- Add geometry-aware modeling now.
- Modify graph features or loss functions while hardening infrastructure.

Why this option was chosen:
- Infrastructure must be stable before scientific model changes can be evaluated fairly.
- It avoids confounding benchmark improvements with architecture changes.

Trade-offs:
- Known modeling limitations remain.
- Publication-level accuracy claims still require future model and baseline work.

Future implications:
- Milestone 2 can evaluate geometry-aware modeling against the frozen benchmark protocol.

## ADR-011: Standardize Official Research Environment Around Python 3.11.9 And PyTorch CUDA 12.1

Decision:
- Downgrade the environment from Python 3.14 to Python 3.11.9 and configure PyTorch 2.3.1+cu121 with PyTorch Geometric 2.8.0. Pin NumPy to version 1.26.4.

Context:
- Python 3.14.3 lacked precompiled CUDA wheels for Windows, forcing CPU fallback.
- Training on CPU was highly computationally intensive, making full multi-seed baseline benchmarking impractical.
- PDB pocket graph construction, RDKit coordinates, and downstream modeling (ESM, MolFormer, Equivariant GNNs) require hardware acceleration and stable packaging.
- NumPy 2.x introduced binary breaks that caused C-extension segmentation faults with PyTorch and PyG.

Alternatives considered:
- Continue executing CPU-only on Python 3.14.
- Standardize on Python 3.10.

Why this option was chosen:
- Python 3.11.9 is highly performant and widely compatible with the PyTorch/PyG CUDA stack.
- Pinning NumPy to 1.26.4 avoids the C-extension binary incompatibility issues.
- Enforcing `CUBLAS_WORKSPACE_CONFIG` inside the seeding logic guarantees complete reproducibility on the GPU.

Trade-offs:
- Required installation of Python 3.11 in user scope and recreating the virtual environment.

Future implications:
- Every future modeling and baseline benchmark run will utilize CUDA GPU acceleration, increasing speed by ~10x-20x.

## ADR-012: Lock and Freeze Optimized Baseline v1.1 and Dataset Graph Cache

Decision:
- Declare the repository frozen at commit `8a90ecaca59d5859191d24badc304efd778eb1f4` with Optimized Baseline v1.1 as the immutable baseline reference. Enforce multi-seed ablation metrics and constitutional baseline certification rules.

Context:
- Transitioning the codebase from a deep learning project into a governed research platform before Phase 3 geometry-aware active modeling.

Alternatives considered:
- Allow fluid code changes on the `main` branch.
- Freeze only the model weights checkpoint while letting codebase files change.

Why this option was chosen:
- It prevents scientific drift and ensures that all future GNN architectures compare against a stable reference point evaluated under identical seed splits and preprocessed graphs.

Trade-offs:
- Code changes to baseline files are blocked except for verified patch bug-fixes, requiring clean git branching for new research.

Future implications:
- Phase 3 features must be implemented in standalone modules and integrated as CLI model selectors, validated by the reproducibility suite before promotion.

