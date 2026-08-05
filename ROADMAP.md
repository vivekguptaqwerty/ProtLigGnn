# ProtLigGNN Long-Term Master Roadmap

Last updated: 2026-07-12

Roadmap status: Phase 2 Complete, entering Phase 3

Primary source documents:
- `BRAIN.md`: living engineering and research knowledge base.
- `PROTLIGGNN_REVERSE_ENGINEERING_REPORT.md`: implementation-first scientific reconstruction.
- `VALIDATION_REPORT.md`: milestone validation record and environment blockers.
- `ARCHITECTURE_DECISIONS.md`: architectural decision records.

This roadmap is the master implementation plan for ProtLigGNN. It is not a task dump. It is the long-term research and engineering plan for turning the repository from a prototype into a reproducible, publication-ready protein-ligand graph learning benchmark and extensible computational drug-discovery research platform.

Implementation remains the source of truth. When this roadmap conflicts with code, artifacts, or validation evidence, the implementation and validation reports win, and this file must be updated.

## Progress Tracker

```text
Phase 0  [##########]  Complete      Reverse Engineering and Knowledge Base
Phase 1  [##########]  Complete      Research Infrastructure
Phase 2  [##########]  Complete      Executable Baseline and Benchmark Freeze
Phase 3  [>---------]  In Progress   Geometry-Aware Modeling
Phase 4  [----------]  Planned       Representation Learning
Phase 5  [----------]  Planned       Interaction Modeling
Phase 6  [----------]  Planned       Explainability
Phase 7  [----------]  Planned       Generative AI
Phase 8  [----------]  Planned       Drug Discovery Platform
Phase 9  [----------]  Planned       Publication Package
```

## Overall Vision

ProtLigGNN should become a scientifically rigorous, reproducible, and extensible research ecosystem for protein-ligand affinity prediction and affinity-guided molecular discovery.

The final system should support:
- Reproducible PDBBind-style affinity benchmarks.
- Deterministic dataset splits with leakage validation.
- Complete dataset, split, configuration, environment, code, checkpoint, metric, and artifact provenance.
- Multi-seed evaluation with confidence intervals and statistical comparisons.
- Frozen baseline protocols before modeling changes.
- Modular geometry-aware and representation-learning experiments.
- Explicit protein-ligand interaction modeling.
- Explainability and error analysis for every prediction.
- Modern target-conditioned molecular generation.
- Publication-ready tables, figures, supplementary materials, dataset cards, model cards, and reproducibility packages.

The project should make conservative scientific claims:
- No architecture improvement is valid until it beats the frozen baseline under identical data, split, seed, metric, and artifact protocols.
- No generative claim is valid until generated molecules, scoring, filtering, docking, and failure counts are fingerprinted and reproducible.
- No publication claim is valid unless it can be traced to a run ID in `experiments/registry.csv`.

## Current Phase

Current active phase:

```text
Phase 2: Executable Baseline and Benchmark Freeze
```

Purpose:
- Convert the completed infrastructure into a clean, executable benchmark baseline.
- Resolve the current Torch/PyG environment blocker.
- Add correctness tests for parsing, graph construction, split handling, model forward passes, metrics, artifact validation, and resume.
- Freeze the official dataset fingerprint, split manifests, benchmark protocol, and baseline metrics before research modeling begins.

Why this phase comes before geometry-aware modeling:
- `VALIDATION_REPORT.md` records that benchmark utilities, split validation, artifact validation, config hashing, registry writing, and comparison utilities pass without Torch.
- Full training and inference validation remain blocked in the current workspace because the available Python interpreter lacks `torch`, and the local `venv` points to a missing Python installation.
- The reverse-engineering report identifies parser coverage, missing tests, unpinned dependencies, missing rich historical artifacts, and random historical splits as major threats to publication validity.

## Phase 0: Reverse Engineering and Knowledge Base

Status:
- Complete.

Completion evidence:
- `PROTLIGGNN_REVERSE_ENGINEERING_REPORT.md`.
- `BRAIN.md`.

### Objectives

- Reverse engineer the complete project from implementation rather than documentation.
- Identify the real research objective, model architecture, data pipeline, training loop, evaluation strategy, experiment artifacts, and technical debt.
- Establish `BRAIN.md` as the living internal engineering and research memory.
- Separate observed implementation facts from strong inferences and hypotheses.
- Prevent future work from overclaiming unsupported novelty or accuracy.

### Deliverables

- Implementation-first reverse-engineering report.
- Living `BRAIN.md` knowledge base.
- Repository structure and execution-flow map.
- Model architecture reconstruction.
- Dataset and parser audit.
- Training, inference, GenAI, GAPS, and Vina workflow reconstruction.
- Publication-readiness assessment.
- Known limitation list.

### Entry Criteria

- Repository available locally.
- Source code, scripts, data layout, outputs, and dependency files readable.
- No assumption that README or generated summaries are authoritative.

### Exit Criteria

- Research objective reconstructed from implementation evidence.
- Core model and data pipeline documented.
- Current limitations and blockers documented.
- Future developers know what to read first.
- BRAIN startup workflow established.

### Dependencies

- Local repository state.
- Source code access.
- Existing output summaries and retained artifacts.

### Risks

- Historical artifacts may be missing, deleted, or not traceable to checkpoints.
- Documentation may describe intended behavior rather than actual behavior.
- Local dataset may differ from a clean clone because data is ignored by Git.

### Success Metrics

- Every major script has a known role.
- Every major research claim is tied to implementation evidence.
- Architecture, training, evaluation, and limitations are captured in durable project documents.
- Known blockers are explicit rather than hidden.

### Estimated Effort

- Completed.
- Original effort class: 1 to 2 research-engineering days.

### What Must NOT Change

- Do not rewrite historical findings to match later aspirations.
- Do not turn hypotheses into facts.
- Do not modify the reverse-engineering report casually; use `BRAIN.md` for living updates.
- Do not claim cross-attention improves performance unless new benchmark evidence proves it.

## Phase 1: Research Infrastructure

Status:
- Complete.

Completion evidence:
- `benchmarking.py`.
- `benchmark.py`.
- `experiment_infra.py`.
- `compare_runs.py`.
- `protliggnn_train.py` infrastructure integration.
- `VALIDATION_REPORT.md`.
- `ARCHITECTURE_DECISIONS.md`.
- Updated `BRAIN.md`.

### Objectives

- Transform the project from ad hoc script outputs into a reproducible experiment framework.
- Preserve backward compatibility with existing training and inference workflows.
- Add deterministic splits, experiment directories, manifests, metrics, plots, error analysis, statistical summaries, and benchmark aggregation.
- Add dataset/config fingerprinting, registry tracking, duplicate detection, resume support, environment capture, structured logs, split validation, artifact validation, reproducibility scoring, and run comparison.
- Keep the GNN architecture, dataset definition, target labels, and research hypothesis unchanged.

### Deliverables

- Deterministic split framework:
  - `train_ids.txt`.
  - `val_ids.txt`.
  - `test_ids.txt`.
  - split `metadata.json`.
  - fixed, seed-generated, scaffold, protein-family, and temporal split modes where metadata permits.
- Experiment directory structure:
  - `experiments/run_001/`, `run_002/`, and later run IDs.
  - config, CLI args, seed, git metadata, split manifest, metrics, predictions, plots, logs, checkpoint, environment, training history, and manifest.
- Multi-seed benchmark runner:
  - `python benchmark.py`.
  - default seeds `42`, `123`, `777`, `2024`, and `3407`.
  - `results_summary.csv`.
  - `benchmark_report.md`.
- Statistical reporting:
  - mean, median, standard deviation, min, max, normal-theory CI95, bootstrap CI95.
  - paired-test placeholder until matched baseline groups are present.
- Expanded evaluation package:
  - RMSE, MAE, Pearson, Spearman, R2, MSE, median absolute error, bias, mean prediction.
  - prediction scatter, residual scatter, error histogram, residual distribution, prediction histogram, calibration curve where meaningful.
- Error analysis:
  - largest errors, best predictions, worst predictions, average prediction, outliers, bias.
  - `error_analysis.csv` and Markdown report.
- Dataset fingerprinting:
  - `dataset_hash.json` with SHA256, file size, and modification timestamps.
  - stable dataset digest for duplicate detection.
- Configuration fingerprinting:
  - canonical config hash.
  - `config_hash.txt`.
  - duplicate experiment warnings and `--allow_duplicate`.
- Experiment registry:
  - `experiments/registry.csv`.
  - run ID, timestamp, git commit, dataset version, dataset hash, config hash, split strategy, seed, validation RMSE, test metrics, status, duration, and checkpoint path.
- Resume support:
  - `--resume latest`.
  - `--resume run_004`.
  - `--resume <checkpoint>`.
  - model, optimizer, optional scheduler, epoch, and RNG state restoration for new checkpoints.
- Structured logging:
  - `experiment.log`.
  - INFO, WARNING, ERROR, DEBUG.
- Environment freeze:
  - `environment.json`.
  - `environment.lock`.
  - `requirements.lock`.
  - `pip_freeze.txt`.
- Split validation:
  - train/validation/test disjointness.
  - duplicate IDs.
  - missing IDs.
  - orphan IDs.
  - `split_validation.json`.
- Artifact validation:
  - checkpoint, history, predictions, metrics, manifest, config, environment, logs, plots, split manifest, dataset hash.
  - `artifact_validation.json`.
- Reproducibility scoring:
  - `reproducibility_report.md`.
  - JSON score artifact.
- Experiment comparison:
  - `python compare_runs.py run_001 run_007`.
  - console table, CSV, and Markdown comparison.
- Architecture decision records:
  - `ARCHITECTURE_DECISIONS.md`.

### Entry Criteria

- Phase 0 completed.
- Research objective and existing architecture understood.
- Infrastructure changes scoped to reproducibility and benchmarking.
- No redesign of the GNN or dataset.

### Exit Criteria

- New experiments are self-contained and uniquely identifiable.
- Dataset, split, config, environment, checkpoint, metric, prediction, log, plot, and manifest artifacts are produced for tracked runs.
- Split leakage is detected before training.
- Duplicate configurations can be detected.
- Multi-seed benchmarking can be launched from one command.
- Experiment comparison works from saved artifacts.
- Documentation and ADRs capture the infrastructure decisions.

Validation caveat:
- Infrastructure utilities were validated in the available Python environment.
- Full Torch/PyG-dependent training and inference validation is deferred to Phase 2 because the current interpreter does not have `torch`, and the local `venv` is non-portable.

### Dependencies

- Phase 0 knowledge base.
- Existing script-based training and inference paths.
- Standard Python libraries for most infrastructure.
- Torch/PyG only for full model execution, not for dry-run utility validation.

### Risks

- Historical experiments cannot be fully reconstructed because many rich artifacts and exact split manifests are missing.
- New checkpoints can resume optimizer/RNG state, but older checkpoints may not contain those fields.
- CSV registry is simple and inspectable but not robust for concurrent multi-process writes.
- Scaffold, protein-family, and temporal split quality depends on metadata availability.

### Success Metrics

- `benchmark.py --dry_run` produces run plans and benchmark artifacts.
- Split validation fails on leakage.
- Artifact validation detects missing required files.
- Config hashes are stable for equivalent scientific configurations.
- Dataset stable digests ignore timestamp-only JSON churn while raw hashes remain available.
- Registry rows can be written and read.
- `compare_runs.py` generates console, CSV, and Markdown outputs.
- `BRAIN.md`, `VALIDATION_REPORT.md`, and `ARCHITECTURE_DECISIONS.md` are synchronized.

### Estimated Effort

- Completed.
- Original effort class: 2 to 4 engineering days for infrastructure implementation and validation under environment constraints.

### What Must NOT Change

- Do not change the model architecture.
- Do not change the target label definition.
- Do not change the dataset without fingerprinting and versioning.
- Do not remove legacy `outputs/` behavior while backward compatibility is required.
- Do not mark an experiment successful without required artifacts.
- Do not silently continue if split leakage is detected.

## Phase 2: Executable Baseline and Benchmark Freeze

Status:
- Complete.

### Objectives

- Reconstruct a clean executable environment.
- Make core training, inference, resume, evaluation, benchmark aggregation, and comparison commands run end to end.
- Fix known parser correctness issues before freezing official baselines.
- Add automated tests and smoke tests around the highest-risk paths.
- Freeze the official dataset fingerprint, split manifests, benchmark protocol, and baseline metrics.
- Establish the baseline that all future modeling phases must beat.

### Deliverables

- Reproducible environment:
  - Python 3.10 or 3.11 environment.
  - compatible Torch, PyG, RDKit, BioPython, SciPy, scikit-learn, NumPy, pandas, tqdm, and Matplotlib.
  - documented install command or environment file.
  - successful `python protliggnn_train.py --help`.
  - successful `python infer_affinity.py --help`.
- Parser correctness:
  - support for `Kd`, `Ki`, `IC50`, `EC50`, and related digit-containing assay names.
  - explicit handling or warning policy for censored labels such as `<`, `>`, `<=`, and `>=`.
  - parser tests covering units `M`, `mM`, `uM`, `nM`, and `pM`.
- Automated tests:
  - config hashing.
  - dataset fingerprinting.
  - split generation and validation.
  - artifact validation.
  - registry writing.
  - comparison utility.
  - affinity parser.
  - graph construction on a tiny fixture.
  - model forward shape on tiny synthetic or fixture graphs.
  - checkpoint load and resume state presence.
- Smoke tests:
  - tiny tracked training run.
  - tiny inference run.
  - tiny benchmark dry run.
  - resume from a new checkpoint.
  - artifact validation failure test.
- Frozen benchmark protocol:
  - official dataset version string.
  - official dataset hash.
  - official split strategy and split manifests.
  - official seeds.
  - official metrics and plotting package.
  - official baseline command lines.
- Frozen baseline artifacts:
  - no-crossgraph baseline.
  - current crossgraph baseline.
  - test metrics with multi-seed summaries.
  - `benchmark_report.md`.
  - completed registry rows.
  - complete experiment manifests.
- Updated documentation:
  - `BRAIN.md`.
  - `VALIDATION_REPORT.md`.
  - roadmap status update when Phase 2 exits.

### Entry Criteria

- Phase 1 infrastructure is implemented.
- Current environment blocker is known and documented.
- Dataset is available locally or a documented dataset acquisition path exists.
- No geometry-aware or representation-learning changes are introduced yet.

### Exit Criteria

- Repository installs into a clean supported environment.
- Training starts and completes on a tiny smoke configuration.
- Dataset discovery and graph construction run successfully.
- Split manifests are generated and validated.
- Checkpoint is saved and reloads.
- Resume works from `latest`, run ID, and checkpoint path for new checkpoints.
- Inference works with a compatible checkpoint.
- Benchmark runner completes at least a tiny multi-seed run.
- Official baseline benchmark is regenerated with complete artifacts.
- Registry, reproducibility reports, artifact validation, plots, predictions, and benchmark reports are present.
- `VALIDATION_REPORT.md` records the end-to-end verification.

### Dependencies

- Phase 1 experiment infrastructure.
- Compatible Torch/PyG/RDKit environment.
- Access to the local PDBBind-style dataset.
- Sufficient CPU/GPU resources for baseline regeneration.

### Risks

- PyG wheel compatibility can be fragile on Windows.
- RDKit install source may need to change from `rdkit-pypi` to conda or another supported channel.
- Parser fixes can change the effective dataset size and invalidate historical metric comparisons.
- Retained historical metrics may not reproduce because original split manifests and rich artifacts were missing.
- CPU-only execution may make full five-seed benchmarks slow.

### Success Metrics

- `python -m pytest` or equivalent test command passes.
- `python protliggnn_train.py --help` passes.
- `python infer_affinity.py --help` passes.
- Tiny training, inference, and benchmark smoke runs complete without runtime errors.
- Full baseline runs produce complete artifact validation.
- Reproducibility score is generated for every baseline run.
- `experiments/registry.csv` contains official baseline entries.
- Baseline `benchmark_report.md` contains mean, median, standard deviation, CI95, bootstrap CI95, best, and worst metrics.

### Estimated Effort

- 3 to 7 engineering days.
- Higher if dependency resolution or full baseline training is CPU-only.

### What Must NOT Change

- Do not redesign the GNN.
- Do not introduce geometry-aware layers.
- Do not change the research objective.
- Do not change datasets without producing a new dataset version and fingerprint.
- Do not compare future models against historical unmanifested metrics as if they were frozen baselines.
- Do not silently treat parser changes as equivalent to old experiments.

## Phase 3: Geometry-Aware Modeling

Status:
- Completed: Phase 3.1A (Geometry Feature Injection - Ligand-Only) successfully benchmarked and certified as statistically equivalent to Baseline v1.1. Reference baseline model v1.1 retained.

### Objectives

- Evaluate whether explicit 3D geometry improves affinity prediction beyond the frozen Phase 2 baseline.
- Introduce geometry-aware models only behind explicit config/CLI options.
- Compare candidate models under the same dataset, splits, seeds, metrics, artifact protocol, and statistical evaluation as the frozen baseline.
- Keep claims focused on measured benchmark improvement, not architectural novelty alone.

### Deliverables

- Geometry-aware candidate implementations:
  - EGNN candidate.
  - PaiNN candidate.
  - TorchMD-Net candidate or carefully scoped TorchMD-Net-style integration.
  - distance-aware message passing variant.
- Improved edge attributes:
  - ligand bond type and bond geometry where available.
  - residue-residue distances.
  - ligand-protein contact distances if an interaction graph is introduced.
- Improved pooling candidates:
  - attention pooling.
  - interaction-site pooling.
  - distance-weighted pooling.
- Config and CLI switches for each model family.
- Shape tests and smoke tests for each candidate.
- Multi-seed benchmark reports against frozen Phase 2 baseline.
- Ablation report separating geometry features, equivariant layers, and pooling changes.

### Entry Criteria

- Phase 2 baseline is frozen.
- Official dataset hash and split manifests exist.
- Official baseline metrics exist with multi-seed summaries.
- Test suite covers existing graph construction and model forward paths.

### Exit Criteria

- Each geometry-aware model runs through the full experiment lifecycle.
- Each model has complete artifacts, registry rows, reproducibility reports, and comparison reports.
- Results are compared against the frozen baseline using matched seeds and splits.
- Statistical summaries show whether improvements are meaningful.
- Any added geometry processing is fingerprinted and versioned.

### Dependencies

- Phase 2 executable baseline.
- Stable graph preprocessing and tests.
- Sufficient compute for heavier models.
- Potential new dependencies for geometry-aware architectures.

### Risks

- Geometry-aware models may be substantially slower and more memory-intensive.
- Bound ligand poses can introduce benchmark-specific assumptions that do not transfer to unbound or generated ligands.
- Implementation complexity can obscure whether gains come from geometry, capacity, or preprocessing differences.
- TorchMD-Net integration may require nontrivial dependency and API compatibility work.

### Success Metrics

- At least one geometry-aware candidate completes the official benchmark protocol.
- Results include RMSE, MAE, Pearson, Spearman, R2, calibration, and error analysis.
- Matched-seed comparisons against baseline are generated.
- Runtime and memory overhead are recorded.
- Claims are limited to statistically supported findings.

### Estimated Effort

- 2 to 4 research weeks.

### What Must NOT Change

- Do not change the frozen Phase 2 dataset, splits, seeds, or metrics for headline comparisons.
- Do not remove the original ProtLigGNN baseline.
- Do not conflate multiple changes without ablations.
- Do not claim state-of-the-art performance without external baseline context.

## Phase 4: Representation Learning

Status:
- Planned.

### Objectives

- Evaluate whether pretrained protein and ligand representations improve affinity prediction over handcrafted graph features.
- Compare handcrafted features, pretrained embeddings, and hybrid models under the frozen benchmark protocol.
- Keep representation upgrades modular and ablation-friendly.

### Deliverables

- Protein representation integrations:
  - ESM-2 embeddings.
  - ProtT5 embeddings.
- Ligand representation integrations:
  - ChemBERTa embeddings.
  - MolFormer embeddings.
  - GROVER embeddings, if practical.
- Feature-cache system for pretrained embeddings:
  - model name.
  - checkpoint/version.
  - tokenizer/version.
  - input sequence/SMILES canonicalization.
  - embedding hash.
- Baselines:
  - handcrafted-only.
  - pretrained-only.
  - handcrafted plus pretrained.
  - frozen embeddings versus fine-tuned embeddings where feasible.
- Tests for embedding cache validity and shape compatibility.
- Multi-seed benchmark reports against Phase 2 and Phase 3 baselines.

### Entry Criteria

- Phase 2 benchmark is frozen.
- Phase 3 either completed or explicitly deferred with documented rationale.
- Dataset identifiers can be mapped reliably to protein sequences and ligand SMILES.
- Compute and storage requirements for embeddings are accepted.

### Exit Criteria

- Protein and ligand embedding pipelines are reproducible.
- Embedding caches are fingerprinted and invalidated correctly.
- Representation-learning variants are benchmarked under identical splits and seeds.
- Results distinguish the value of protein embeddings, ligand embeddings, and hybrid features.

### Dependencies

- Frozen benchmark protocol.
- Sequence and SMILES extraction correctness.
- Additional pretrained model dependencies.
- Cache management from earlier infrastructure.

### Risks

- Pretrained models may be large and slow to run.
- Protein sequence extraction from PDB structures may be incomplete or inconsistent.
- SMILES canonicalization choices can affect ligand embedding reproducibility.
- Fine-tuning large models may exceed available compute.

### Success Metrics

- Embedding cache artifacts are reproducible from config.
- Every embedding run has complete dataset/config/environment fingerprints.
- Representation variants are compared with confidence intervals.
- Reports identify whether pretrained embeddings outperform handcrafted features.

### Estimated Effort

- 2 to 5 research weeks.

### What Must NOT Change

- Do not replace the benchmark protocol.
- Do not mix pretrained embeddings into the baseline without explicit ablation labels.
- Do not allow embedding cache drift without a new fingerprint.
- Do not use undocumented pretrained model checkpoints.

## Phase 5: Interaction Modeling

Status:
- Planned.

### Objectives

- Move beyond separate ligand and protein graph encoders by explicitly modeling protein-ligand interactions.
- Evaluate interaction edges and contact mechanisms through controlled ablations.
- Test whether chemically meaningful interaction graphs improve affinity prediction and interpretability.

### Deliverables

- Explicit protein-ligand interaction graph.
- Interaction edge types:
  - hydrogen bond edges.
  - salt bridge edges.
  - pi-pi interaction edges.
  - hydrophobic interaction edges.
  - metal coordination edges where relevant.
  - generic distance-aware contact edges.
- Interaction edge attributes:
  - distance.
  - angle where meaningful.
  - donor/acceptor roles.
  - residue and atom type context.
- Interaction message-passing layer or bipartite graph module.
- Ablations:
  - distance-only.
  - chemistry-only.
  - distance plus chemistry.
  - interaction graph plus original cross-attention.
  - interaction graph without cross-attention.
- Interaction extraction report with skipped/failure counts.
- Benchmark and explainability hooks for interaction edges.

### Entry Criteria

- Phase 2 baseline is frozen.
- Geometry and representation phases are complete or intentionally sequenced around this phase.
- Protein and ligand coordinates are validated and consistently available.
- Tests exist for graph construction and interaction extraction.

### Exit Criteria

- Interaction graphs are generated reproducibly.
- Interaction graph artifacts are fingerprinted.
- Each interaction mechanism is evaluated through ablation.
- Reports identify which interaction types contribute to performance or explanation quality.
- Failure cases are logged rather than silently skipped.

### Dependencies

- RDKit and BioPython coordinate reliability.
- Possibly additional chemistry libraries for robust interaction detection.
- Frozen benchmark protocol.
- Geometry-aware infrastructure if distances and angles become central.

### Risks

- Rule-based interaction detection can be brittle.
- Bound crystal structures may encode target leakage relative to prospective docking contexts.
- Added interaction features may improve in-distribution metrics without improving generalization.
- Complex ablations can multiply experiment cost.

### Success Metrics

- Interaction extraction coverage is reported.
- No silent skipped-interaction failures.
- Ablation reports show mechanism-level contributions.
- Matched-seed comparisons quantify performance and uncertainty.
- Interaction maps can be consumed by explainability tools in Phase 6.

### Estimated Effort

- 3 to 6 research weeks.

### What Must NOT Change

- Do not change the frozen split protocol for headline comparisons.
- Do not introduce interaction features without ablations.
- Do not claim biochemical mechanism discovery from predictive edges alone.
- Do not silently drop complexes that fail interaction extraction without reporting dataset impact.

## Phase 6: Explainability

Status:
- Planned.

### Objectives

- Provide a reproducible explanation pipeline for affinity predictions.
- Help researchers inspect whether model evidence aligns with chemically plausible interactions.
- Support publication-quality interpretability figures and failure analysis.

### Deliverables

- Explanation methods:
  - GNNExplainer.
  - Integrated Gradients.
  - attention visualization where attention weights are available.
  - atom and residue saliency maps.
  - interaction-edge saliency when Phase 5 modules exist.
  - counterfactual explanations.
- Explanation artifacts:
  - per-prediction explanation JSON.
  - atom-level scores.
  - residue-level scores.
  - edge-level scores.
  - rendered ligand and pocket visualizations.
  - top explanation summaries for best and worst predictions.
- Explanation reports:
  - explanation stability across seeds.
  - explanation overlap with known or inferred contacts.
  - failure-mode explanations for top residuals.
- Tests and smoke examples for explanation artifact generation.

### Entry Criteria

- A stable baseline or improved model exists.
- Checkpoints and predictions are reproducible.
- Model forward paths expose the tensors required by attribution methods.
- If attention is used as evidence, attention weights must be saved rather than discarded.

### Exit Criteria

- Every benchmark prediction can be traced to an explanation pipeline.
- Explanations are reproducibly generated from saved checkpoints and inputs.
- Explanation artifacts are included in experiment manifests.
- Reports distinguish explanation as analysis from explanation as proof of mechanism.

### Dependencies

- Stable model and checkpoint format.
- Interaction modeling, if interaction-level explanations are required.
- Visualization dependencies.
- Compute budget for attribution methods.

### Risks

- Attention weights may not be faithful explanations.
- Attribution methods can be unstable across seeds.
- Counterfactual generation can accidentally become a modeling or generation project.
- Explanation visuals can be misleading if not tied to quantitative artifacts.

### Success Metrics

- Explanation pipeline runs on a representative sample and on selected top-error cases.
- Per-prediction explanation files are manifest-tracked.
- Explanation stability metrics are reported.
- Publication figures can be regenerated from explanation artifacts.

### Estimated Effort

- 2 to 4 research weeks.

### What Must NOT Change

- Do not present explanations as causal proof.
- Do not change prediction metrics or splits to favor explainability.
- Do not rely on attention visualizations unless attention weights are actually captured.
- Do not omit failed explanations from reports.

## Phase 7: Generative AI

Status:
- Planned.

### Objectives

- Replace or augment the current character-level SMILES LSTM with modern molecular generation methods.
- Make molecular generation target-aware, reproducible, and benchmarkable.
- Evaluate generated molecules using the established affinity, filtering, docking, and provenance infrastructure.

### Deliverables

- Modern generator candidates:
  - MolGPT-style SMILES transformer.
  - diffusion-based molecular generation candidate.
  - pocket-conditioned generation.
  - reinforcement-learning optimization loop.
- Conditioning inputs:
  - protein pocket representation.
  - affinity predictor feedback.
  - optional interaction or geometry features from earlier phases.
- Generation benchmark:
  - validity.
  - uniqueness.
  - novelty.
  - QED.
  - Lipinski compliance.
  - diversity.
  - predicted affinity.
  - docking score where available.
  - synthetic accessibility or medicinal chemistry filters if added.
- Reproducible generation artifacts:
  - generator config.
  - seed.
  - checkpoint.
  - vocabulary/tokenizer.
  - generated molecule file.
  - failed molecule counts.
  - filter report.
  - scoring report.
  - docking report.
- Comparison against current character-level LSTM baseline.

### Entry Criteria

- Phase 2 baseline and scoring infrastructure are reproducible.
- Generated molecule workflow is traceable and fingerprinted.
- Affinity predictor checkpoint used for scoring is registry-linked.
- Target protein input is versioned and documented.

### Exit Criteria

- At least one modern generator runs end to end.
- Generated molecules are reproducible from seed and config.
- Scoring, filtering, and docking artifacts are complete.
- Generator results are compared against the LSTM baseline and against simple sampling baselines.
- Claims avoid implying experimental validation where only computational screening exists.

### Dependencies

- Stable affinity predictor.
- Scoring and GAPS workflow.
- Optional docking tool availability.
- Potential new deep generative modeling dependencies.

### Risks

- Generated molecules may optimize model artifacts rather than true binding.
- RL optimization can exploit weaknesses in the affinity predictor.
- Docking and generated affinity scores are not experimental evidence.
- Large generators can increase compute and dependency complexity.

### Success Metrics

- Generation runs are fully reproducible from artifacts.
- Validity, uniqueness, novelty, diversity, QED, predicted affinity, and docking summaries are produced.
- Top molecules include failure and filtering provenance.
- Modern generator is benchmarked against the old LSTM under identical evaluation criteria.

### Estimated Effort

- 4 to 8 research weeks.

### What Must NOT Change

- Do not replace the affinity benchmark with generation metrics.
- Do not hide invalid or failed molecules.
- Do not optimize GAPS weights without sensitivity analysis or a documented objective.
- Do not claim generated molecules are drug candidates without stronger validation.

## Phase 8: Drug Discovery Platform

Status:
- Planned.

### Objectives

- Integrate validated modules into a modular computational drug-discovery research platform.
- Support affinity prediction, docking, molecule generation, ADMET/toxicity screening, virtual screening, explainability, and experiment dashboards.
- Preserve the benchmark core while enabling applied workflows.

### Deliverables

- Platform modules:
  - affinity prediction.
  - docking workflow.
  - molecular generation.
  - ADMET prediction integration.
  - toxicity prediction integration.
  - virtual screening pipeline.
  - explainability pipeline.
  - experiment dashboard or report viewer.
- Workflow orchestration:
  - target setup.
  - ligand library ingestion.
  - batch scoring.
  - filtering.
  - docking.
  - ranking.
  - explanation.
  - report generation.
- Data and artifact contracts:
  - target card.
  - ligand library card.
  - model card.
  - dataset card.
  - screening run manifest.
- Usability improvements:
  - consistent CLI commands.
  - clear config files.
  - reproducible examples.
  - documented output schema.

### Entry Criteria

- Core benchmark is stable.
- Modeling improvements are benchmarked and documented.
- Generation and explainability workflows are reproducible.
- Major scripts have tests or smoke validation.

### Exit Criteria

- Users can run an end-to-end virtual screening workflow from documented commands.
- Every screening run produces a complete manifest.
- Platform modules can be used independently or together.
- Dashboard/report artifacts expose metrics, rankings, explanations, and provenance.
- Research benchmark and applied platform modes remain clearly separated.

### Dependencies

- Phases 2 through 7.
- External tools for docking and optional ADMET/toxicity prediction.
- More formal package or module organization may be required.

### Risks

- Platform scope can sprawl beyond research goals.
- External ADMET/toxicity tools may introduce licensing or reproducibility issues.
- Dashboard work can distract from scientific validation.
- Applied workflows may invite stronger claims than the evidence supports.

### Success Metrics

- End-to-end virtual screening command completes on a small target/library example.
- Affinity, generation, docking, ADMET/toxicity, ranking, and explanation outputs are linked in one manifest.
- Module-level outputs remain reproducible and independently testable.
- Documentation clearly separates benchmark evidence from exploratory screening.

### Estimated Effort

- 6 to 12 engineering/research weeks.

### What Must NOT Change

- Do not weaken benchmark rigor to make platform demos easier.
- Do not merge exploratory platform results into official benchmark claims.
- Do not introduce external services without documenting availability and reproducibility impact.
- Do not hide tool failures in screening summaries.

## Phase 9: Publication Package

Status:
- Planned.

### Objectives

- Generate publication-ready research artifacts from validated experiment outputs.
- Make tables, figures, supplementary materials, model cards, dataset cards, and reproducibility packages traceable to registry run IDs.
- Support paper writing without manual metric copying or unsupported claims.

### Deliverables

- `python build_publication_package.py`.
- Auto-generated paper assets:
  - main result tables.
  - ablation tables.
  - benchmark comparison tables.
  - confidence interval tables.
  - training curves.
  - prediction scatter plots.
  - residual plots.
  - calibration plots.
  - explanation figures.
  - generation and docking summaries.
- Supplementary package:
  - full configs.
  - split manifests.
  - dataset fingerprints.
  - environment locks.
  - registry snapshot.
  - reproducibility reports.
  - artifact validation reports.
  - model cards.
  - dataset cards.
  - experiment summaries.
- Claim audit:
  - every numerical result maps to a run ID.
  - every figure maps to source artifacts.
  - every unsupported claim is flagged.
- Release checklist:
  - clean install instructions.
  - benchmark reproduction commands.
  - expected hashes.
  - known limitations.

### Entry Criteria

- Official baselines and selected improved models are complete.
- Experiment registry is accurate.
- Required plots, predictions, metrics, and manifests exist.
- Documentation is synchronized.
- Publication claims have been reviewed against implementation evidence.

### Exit Criteria

- Running `python build_publication_package.py` produces a publication-ready package.
- All paper tables and figures are regenerated from validated artifacts.
- Supplementary materials are complete.
- Dataset and model cards are present.
- Reproducibility package can be audited without hidden local state.
- Final claims match benchmark evidence.

### Dependencies

- Phases 2 through 8 as needed for the intended paper scope.
- Complete registry and artifact manifests.
- Stable plotting/report-generation scripts.

### Risks

- Results may be weaker under stricter splits than historical summaries.
- Some external baselines may be hard to reproduce.
- Publication timelines may pressure premature claims.
- Manual edits to generated assets can break traceability.

### Success Metrics

- Zero untraceable numerical claims in the package.
- Every figure/table has source artifact metadata.
- Reproduction commands are present and tested.
- Reproducibility package includes environment, split, config, checkpoint, metrics, predictions, and logs.
- Reviewer-facing limitations are explicit and evidence-grounded.

### Estimated Effort

- 2 to 4 research weeks after final experiments are complete.

### What Must NOT Change

- Do not manually edit generated metrics.
- Do not cite historical retained metrics as official unless they are regenerated or clearly labeled historical.
- Do not claim state-of-the-art performance without fair baselines.
- Do not hide negative or non-significant findings.

## Dependency Map

```text
Phase 0: Reverse Engineering and Knowledge Base
    |
    v
Phase 1: Research Infrastructure
    |
    v
Phase 2: Executable Baseline and Benchmark Freeze
    |
    +--> Phase 3: Geometry-Aware Modeling
    |       |
    |       v
    |   Phase 5: Interaction Modeling
    |       |
    |       v
    |   Phase 6: Explainability
    |
    +--> Phase 4: Representation Learning
    |       |
    |       v
    |   Phase 5: Interaction Modeling
    |
    +--> Phase 7: Generative AI
            |
            v
Phase 8: Drug Discovery Platform
    |
    v
Phase 9: Publication Package
```

Sequencing rule:
- Phase 2 is the gate for all research modeling phases.
- Phases 3 and 4 can proceed in parallel only after Phase 2 exits.
- Phase 5 depends on stable geometry and/or representation choices.
- Phase 6 depends on stable models and saved explanation-relevant tensors.
- Phase 7 can begin after Phase 2 if it uses the frozen affinity baseline, but stronger target-conditioned generation benefits from Phases 3 through 5.
- Phase 8 should not begin until core research workflows are reproducible.
- Phase 9 should only package validated artifacts.

## Global Success Criteria

The project is on track if every completed phase preserves:
- Backward-compatible CLI behavior unless a breaking change is explicitly approved and documented.
- Complete experiment directories for tracked runs.
- Registry entries for tracked runs.
- Dataset fingerprints.
- Config hashes.
- Split manifests and split validation.
- Environment locks.
- Structured logs.
- Checkpoints.
- Predictions.
- Metrics.
- Plots.
- Artifact validation.
- Reproducibility reports.
- Documentation updates in `BRAIN.md`.
- ADR updates for significant architectural decisions.

The project is publication-ready only when:
- Official results are regenerated under the frozen benchmark protocol.
- Metrics include multi-seed uncertainty.
- Baselines are fair and explicit.
- Dataset and split leakage risks are addressed.
- Parser behavior is tested and documented.
- Environment setup is reproducible.
- All reported numbers trace to artifacts.

## Global Risks and Assumptions

### Technical Risks

- The current local Python environment has been standardized with Torch and CUDA enabled.
- The `venv` is standardized using Python 3.11.9 and PyTorch 2.3.1+cu121.
- PyTorch Geometric and RDKit have been successfully installed and verified on Windows with CUDA.
- Dependencies are pinned in OFFICIAL_ENVIRONMENT.md.
- The project is still script-based rather than package-based.
- Graph preprocessing caching has been implemented and validated.
- CSV registry is not ideal for concurrent experiment writers.

### Scientific Risks

- Historical random splits may overestimate generalization.
- Parser changes can materially change the dataset and therefore the benchmark.
- The retained best historical result is no-crossgraph, limiting claims about cross-attention.
- Current protein graph construction ignores explicit edge distances after graph creation.
- Current ligand graph ignores bond type edge attributes.
- Current GenAI workflow is target-agnostic and lightweight.
- Vina validation is computational screening, not experimental validation.

### Artifact Risks

- Historical rich artifacts are missing or deleted in the visible output tree.
- Historical checkpoints may not include optimizer or RNG state.
- Final summaries may not be exactly reproducible from current local artifacts.
- Data under `data/` may not be tracked in Git and may differ across machines.

### Assumptions

- Development continues from repository root `D:\ProtLigGnn`.
- Local PDBBind-style data remains under `data/pdbbind2020`.
- New experiments use the Phase 1 experiment infrastructure.
- Python 3.10 or 3.11 will be used for stable Torch/PyG/RDKit support.
- Modeling improvements wait until Phase 2 freezes the benchmark.
- Publication claims remain conservative unless supported by regenerated artifacts.

## Update Rule

Whenever a phase is completed:

1. Update the progress tracker at the top of this file.
2. Mark the phase status as complete.
3. Record the completion date.
4. Mark completed deliverables explicitly.
5. Link the validation report or benchmark report that proves completion.
6. Update `BRAIN.md` so the living project memory matches this roadmap.
7. Add or update ADRs in `ARCHITECTURE_DECISIONS.md` for significant infrastructure, modeling, data, or publication decisions.
8. Record known limitations that remain after the phase.
9. Do not delete historical caveats; supersede them with dated evidence.

Roadmap synchronization rule:
- `ROADMAP.md` defines planned direction.
- `BRAIN.md` defines current operational truth.
- `VALIDATION_REPORT.md` records what has actually been verified.
- `ARCHITECTURE_DECISIONS.md` records why significant choices were made.
- `PROTLIGGNN_REVERSE_ENGINEERING_REPORT.md` remains the implementation-first historical reconstruction and should not be edited casually.

## Final Vision

The long-term goal is for ProtLigGNN to stand as:

- a reproducible research benchmark for protein-ligand affinity prediction;
- a modular computational drug-discovery platform;
- a publication-ready framework with traceable metrics, artifacts, figures, and claims;
- a long-term extensible graph neural network research ecosystem.

In its mature form, a researcher should be able to clone the repository, install the documented environment, reproduce the frozen baseline, compare new models under identical protocols, inspect every dataset/config/split/checkpoint/metric artifact, generate publication materials automatically, and extend the system toward geometry-aware modeling, representation learning, interaction modeling, explainability, molecular generation, and virtual screening without breaking scientific traceability.
