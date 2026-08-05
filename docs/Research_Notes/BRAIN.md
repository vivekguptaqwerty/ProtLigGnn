# BRAIN.md

Last updated: 2026-07-11 20:03 IST
Project root: `D:\ProtLigGnn`
Primary architectural reference: `PROTLIGGNN_REVERSE_ENGINEERING_REPORT.md`

This file is the living engineering brain for ProtLigGNN. It is not a README. It is the internal technical memory a new engineer should read before making any change.

## 0. Mandatory Startup Workflow

Before any implementation, bug fix, refactor, experiment, or architectural change:

```text
Locate BRAIN.md
    |
    v
Read BRAIN.md completely
    |
    v
Read PROTLIGGNN_REVERSE_ENGINEERING_REPORT.md
    |
    v
Inspect current implementation
    |
    v
Compare code, BRAIN.md, and reverse-engineering report
    |
    v
Plan and implement changes
    |
    v
Validate behavior
    |
    v
Update affected BRAIN.md sections
```

Rules:
- Prefer implementation evidence over README or generated prose.
- Keep historical records. Do not delete experiment history just because newer results exist.
- Update only affected sections after a change.
- Clearly separate current implementation, limitations, planned improvements, and historical decisions.

## 1. Project Overview

ProtLigGNN is a Python research prototype for protein-ligand binding affinity prediction and affinity-guided ligand prioritization.

Current implementation:
- Trains a graph neural network affinity regressor on local PDBBind-style protein-ligand complexes.
- Converts ligands into RDKit atom graphs.
- Converts proteins into ligand-centered residue pocket graphs.
- Encodes ligand graphs with GAT layers and protein pocket graphs with GCN layers.
- Optionally applies bidirectional ligand-protein cross-attention.
- Predicts a scalar pAffinity-like value with an MLP regression head.
- Provides a lightweight generative chemistry loop based on a character-level SMILES LSTM.
- Filters generated molecules with RDKit descriptors.
- Scores generated molecules with ProtLigGNN.
- Ranks generated molecules with GAPS, a hand-weighted multi-objective score.
- Performs secondary validation with Open Babel and AutoDock Vina.

Target users:
- ML researchers working on protein-ligand affinity prediction.
- Drug-discovery engineers prototyping virtual screening workflows.
- Engineers maintaining or extending the repository.
- Technical writers preparing methodology or internal reports.

Scientific objective:
- Determine whether paired ligand molecular graphs and protein pocket residue graphs can predict PDBBind binding affinity well enough to support candidate prioritization.

Business or applied objective:
- Enable computational screening and prioritization of generated ligands before expensive docking, simulation, synthesis, or assay work.

Current maturity:
- Research prototype with a substantially stronger experiment infrastructure layer.
- Coherent end-to-end pipeline exists.
- New experiments now have deterministic splits, registry rows, fingerprints, logs, manifests, artifact validation, and reproducibility reports.
- Not yet publication-grade or production-grade because dependency pinning, tests, parser correctness, baselines, and clean-environment reruns remain unresolved.

## 2. High-Level Architecture

```text
PDBBind index + local complex folders
    |
    v
Label parser: Kd/Ki token -> -log10(molar concentration)
    |
    v
Ligand file                 Protein PDB
SDF/MOL2                    full receptor structure
    |                         |
    v                         v
RDKit Mol                 BioPython PDBParser
    |                         |
    v                         v
Atom graph                Residues within 6A of ligand atoms
    |                         |
    v                         v
Ligand GAT encoder        Protein GCN encoder
    |                         |
    +----------+--------------+
               |
               v
Optional bidirectional cross-attention
               |
               v
Global mean pooling
               |
               v
MLP regression head
               |
               v
Predicted affinity
```

Full research workflow:

```text
Dataset
  -> preprocessing
  -> graph construction
  -> model training
  -> validation and early stopping
  -> test metrics
  -> inference
  -> SMILES extraction
  -> SMILES generator training
  -> molecule generation and filtering
  -> ProtLigGNN scoring
  -> GAPS ranking
  -> Vina docking validation
  -> report package
```

## 3. Repository Structure

```text
D:\ProtLigGnn
|-- BRAIN.md
|-- PROTLIGGNN_REVERSE_ENGINEERING_REPORT.md
|-- README.md
|-- requirements.txt
|-- protliggnn_train.py
|-- infer_affinity.py
|-- genai_extract_ligands.py
|-- train_smiles_generator.py
|-- generate_and_filter_molecules.py
|-- score_generated_with_protliggnn.py
|-- vina_validation.py
|-- dock_top_candidates.py
|-- top_affinity_predictions.py
|-- report_assets.py
|-- generate_paper_figures.py
|-- final_research_package.py
|-- data/
|   `-- pdbbind2020/
|       |-- index/
|       `-- pdbbind_subset/2011-2019/<pdb_id>/
|-- outputs/
|   `-- final_research_package/
|-- venv/
`-- __pycache__/
```

Folders:

| Path | Purpose | Current Notes |
|---|---|---|
| `data/pdbbind2020/index` | PDBBind index files and metadata. | `INDEX_general_PL.2020R1.lst` is the only label file used by training. |
| `data/pdbbind2020/pdbbind_subset/2011-2019` | Local complex structure files. | Contains 12,235 complex dirs in this workspace. Each expected dir has protein, pocket, SDF, and MOL2 files. |
| `outputs/final_research_package` | Compact retained result summaries. | Richer experiment artifacts referenced by scripts are mostly absent or deleted in current worktree. |
| `venv` | Local virtual environment. | Non-portable. `pyvenv.cfg` points to a missing Python 3.10 install under `C:\Users\Admin`. |
| `__pycache__` | Python bytecode cache. | Generated artifact. |

## 4. Complete Feature Inventory

| Feature | Status | Purpose | Main Files | Limitations |
|---|---|---|---|---|
| PDBBind affinity parsing | Implemented | Convert binding tokens to pAffinity labels. | `protliggnn_train.py` | Regex skips labels with digits such as `IC50`; censored labels are treated as exact. |
| Ligand graph construction | Implemented | Convert SDF/MOL2 ligand to atom graph. | `protliggnn_train.py` | No bond type or bond distance edge attributes. |
| Protein pocket graph construction | Implemented | Build residue graph near ligand. | `protliggnn_train.py` | Rebuilds from protein PDB; does not use provided pocket file; no edge distance attributes. |
| Affinity model | Implemented | Predict scalar binding affinity. | `protliggnn_train.py` | Mean pooling and coarse residue features can miss interaction motifs. |
| Cross-attention interaction | Implemented | Condition ligand atoms and residues on opposite graph. | `protliggnn_train.py` | Retained results do not show it winning; attention maps discarded. |
| No-crossgraph ablation | Implemented | Controlled ablation preserving tensor shape. | `protliggnn_train.py` | Currently best retained run is no-crossgraph, which weakens cross-attention novelty. |
| Training loop | Implemented | Optimize affinity model. | `protliggnn_train.py` | No scheduler, AMP, or DDP; new checkpoints include optimizer/RNG state; graph cache is fully implemented. |
| Evaluation metrics | Implemented | MSE, RMSE, MAE, median AE, Pearson, Spearman, R2, bias, and mean prediction. | `protliggnn_train.py`, `benchmarking.py` | Historical retained metrics lack multi-seed statistics until regenerated. |
| Single-pair inference | Implemented | Predict affinity for one protein-ligand pair. | `infer_affinity.py` | Requires trained checkpoint and compatible dependencies. |
| SMILES extraction | Implemented | Build training corpus for molecule generator. | `genai_extract_ligands.py` | Extracts from all local SDF/MOL2; target-agnostic. |
| SMILES LSTM generation | Implemented | Generate candidate molecules. | `train_smiles_generator.py`, `generate_and_filter_molecules.py` | Character-level, CPU-only, target-agnostic, low novelty versus modern generators. |
| RDKit filtering | Implemented | Validate and characterize generated molecules. | `generate_and_filter_molecules.py` | Basic descriptors only; no PAINS, SA score, toxicity, reactivity, or retrosynthesis filters. |
| Generated molecule scoring | Implemented | Score generated molecules using ProtLigGNN. | `score_generated_with_protliggnn.py` | Embeds one molecule at a time; can skip failed 3D embeddings. |
| GAPS ranking | Implemented | Rank generated candidates by affinity, novelty, QED, diversity. | `score_generated_with_protliggnn.py` | Hand-weighted and not statistically validated. |
| AutoDock Vina validation | Implemented | Secondary computational validation. | `vina_validation.py`, `dock_top_candidates.py` | Requires external tools; blind docking box is approximate. |
| Report assets | Implemented | Build tables, plots, final summaries. | `report_assets.py`, `generate_paper_figures.py`, `final_research_package.py` | Depends on local artifacts that may be absent. |
| Experiment infrastructure | Implemented | Fingerprint data/configs, maintain registry, validate splits/artifacts, freeze environments, score reproducibility, resolve resume targets. | `experiment_infra.py`, `protliggnn_train.py` | Full training validation is still blocked in this workspace by missing Torch. |
| Experiment comparison | Implemented | Compare saved run artifacts, metrics, configs, splits, training curves, runtime, and environments. | `compare_runs.py` | Single-run comparisons are descriptive unless backed by multi-seed distributions. |
| Explainability | Planned | Interpret model decisions and attention/contact evidence. | None yet | Attention weights disabled; no saliency or attribution. |
| Tests | Missing | Guard correctness and reproducibility. | None | High priority. |
| Packaging | Missing | Make repo installable and maintainable. | None | Scripts only; no package layout. |

## 5. Research Objective

Problem statement:
- Given a protein receptor structure and ligand structure, predict binding affinity on a continuous pAffinity-like scale.
- Use the trained predictor to prioritize known and generated ligands.

Scientific hypothesis:
- A ligand atom graph plus a ligand-centered protein pocket residue graph encodes enough structure and chemistry to learn affinity trends from PDBBind.
- The trained predictor can act as a useful scoring function inside a generated-ligand prioritization loop.

Research contribution currently supported by implementation:
- An integrated closed-loop workflow that combines graph affinity prediction, lightweight ligand generation, RDKit filtering, GAPS ranking, and Vina validation.

Novelty boundaries:
- Novelty is integration and workflow, not RDKit, Vina, GAT, GCN, LSTM, or standard metrics.
- Cross-attention is implemented, but retained results show no-crossgraph best, so do not claim cross-attention improves performance without fresh evidence.

## 6. End-to-End Execution Flow

Training flow:

```text
python protliggnn_train.py
    |
    v
parse args
    |
    v
set Python, NumPy, Torch seeds
    |
    v
discover PDBBind records
    |
    v
eagerly process records into PyG Data pairs
    |
    v
random 80/10/10 split
    |
    v
train ProtLigGNN with MSE + Adam
    |
    v
validate each epoch and early-stop by validation RMSE
    |
    v
save best checkpoint
    |
    v
reload best checkpoint
    |
    v
test and save CSV/PNG artifacts
```

Generated molecule flow:

```text
genai_extract_ligands.py
    -> outputs/genai/pdbbind_ligands_smiles.csv
train_smiles_generator.py
    -> outputs/genai/smiles_lstm_generator.pt
generate_and_filter_molecules.py
    -> outputs/genai/generated_molecules.csv
score_generated_with_protliggnn.py
    -> outputs/genai/generated_molecules_scored.csv
    -> outputs/genai/top10_generated_candidates.csv
    -> outputs/genai/top10_generated_candidates_by_gaps.csv
dock_top_candidates.py or vina_validation.py
    -> outputs/vina_validation/*
final_research_package.py
    -> outputs/final_research_package/*
```

## 7. Data Pipeline

Dataset:
- Local PDBBind-style data under `data/pdbbind2020`.
- Index file: `data/pdbbind2020/index/INDEX_general_PL.2020R1.lst`.
- Structure files: `data/pdbbind2020/pdbbind_subset/2011-2019/<pdb_id>/`.

Label parsing:
- `parse_affinity_to_pkd` converts tokens like `Kd=49uM` and `Ki=0.43uM` to `-log10(molar)`.
- Current parser accepts only alphabetic assay names, so labels such as `IC50` are skipped.
- Local implementation-derived count from the reverse-engineering pass: 7,335 parseable local labels out of 12,235 local complex directories.

Loading:
- `discover_complex_records` intersects label index entries with discovered protein files.
- `select_ligand_file` prefers `<pdb_id>_ligand.sdf`, then `<pdb_id>_ligand.mol2`, then fallback ligand patterns.
- `load_ligand_molecule` requires a readable RDKit molecule with at least one conformer.

Graph construction:
- Ligand nodes: atoms, 78-dimensional hand-engineered features.
- Ligand edges: directed covalent bonds.
- Protein nodes: standard amino acid residues within 6A of ligand atoms.
- Protein edges: residue center distance <= 8A, with sequential fallback when no contact edges exist.

Splitting:
- New runs use `build_split_manifest` over processed PDB IDs.
- Default split is deterministic seed-based train/val/test.
- Fixed, scaffold, protein-family, and temporal modes are supported where required metadata is available.
- Every run writes split IDs and metadata, then copies the exact manifest into the experiment directory.

Caching:
- None. Graphs are rebuilt and stored in memory each run.

Augmentation:
- None in affinity training.
- Generated molecules use RDKit ETKDG/UFF for 3D conformer generation during scoring and docking.

## 8. Model Architecture

Main model: `ProtLigGNN`.

```text
Ligand x: [num_ligand_atoms, 78]
Protein x: [num_pocket_residues, 30]

LigandEncoder:
  GATConv 78 -> 256, heads=4
  LayerNorm + ELU
  GATConv 256 -> 256, heads=4
  LayerNorm + ELU
  GATConv 256 -> 256, heads=1, concat=False
  LayerNorm + ELU

ProteinEncoder:
  GCNConv 30 -> 256
  LayerNorm + ReLU
  GCNConv 256 -> 256
  LayerNorm + ReLU
  GCNConv 256 -> 256
  LayerNorm + ReLU

Cross interaction:
  ligand queries protein by MultiheadAttention
  protein queries ligand by MultiheadAttention
  residual + LayerNorm
  concat original embedding with context embedding

Pooling:
  global_mean_pool ligand nodes -> [batch, 512]
  global_mean_pool protein nodes -> [batch, 512]
  concat -> [batch, 1024]

Regressor:
  Linear 1024 -> 512
  ReLU
  Dropout 0.2
  Linear 512 -> 128
  ReLU
  Dropout 0.1
  Linear 128 -> 1
```

No-crossgraph mode:
- `--no_crossgraph` concatenates the encoded node embeddings with zeros instead of attention context.
- This keeps the regressor input size unchanged and supports a fair ablation of the interaction module.

Forward pass:
```text
ligand_batch, protein_batch
    -> ligand_encoder, protein_encoder
    -> per-complex cross_graph_interaction
    -> concatenate per-complex chunks
    -> global_mean_pool
    -> concatenate ligand and protein pooled vectors
    -> regressor
    -> predicted affinity [batch]
```

Loss:
- Affinity model uses MSE: `mean((predicted - true)^2)`.

Optimizer:
- Adam, default lr `1e-3`, weight decay `1e-5`.

Metrics:
- MSE, RMSE, MAE, median absolute error, Pearson, Spearman, R2, bias, and mean prediction for individual runs.
- Mean, median, standard deviation, min, max, normal-theory CI95, and bootstrap CI95 for multi-seed benchmarks.

## 9. Mathematical Foundation

Label:
```text
c = value * unit_scale
y = -log10(c)
```

Ligand graph:
```text
G_l = (V_l, E_l)
x_a in R^78 for each atom a
E_l = directed covalent bonds from RDKit
```

Protein pocket graph:
```text
G_p = (V_p, E_p)
V_p = residues with min distance to ligand atoms <= 6A
x_r in R^30 for each residue r
E_p = residue pairs with center distance <= 8A
```

Encoders:
```text
H_l^k = LN(ELU(GAT_k(H_l^{k-1}, E_l)))
H_p^k = LN(ReLU(GCN_k(H_p^{k-1}, E_p)))
```

Cross-attention:
```text
C_l = LN(MHA(Q=H_l, K=H_p, V=H_p) + H_l)
C_p = LN(MHA(Q=H_p, K=H_l, V=H_l) + H_p)
Z_l = concat(H_l, C_l)
Z_p = concat(H_p, C_p)
```

No-crossgraph:
```text
Z_l = concat(H_l, 0)
Z_p = concat(H_p, 0)
```

Readout and prediction:
```text
g_l = mean_nodes(Z_l)
g_p = mean_nodes(Z_p)
h = concat(g_l, g_p)
y_hat = MLP(h)
```

Training objective:
```text
L = (1 / B) * sum_i (y_hat_i - y_i)^2
```

GAPS:
```text
GAPS = 0.50 * minmax(predicted_affinity)
     + 0.20 * novelty
     + 0.15 * minmax(QED)
     + 0.15 * diversity

diversity_i = 1 - max_{j != i} Tanimoto(MorganFP_i, MorganFP_j)
```

## 10. Module Documentation

### `protliggnn_train.py`

Purpose:
- Core implementation for data discovery, graph construction, model definition, training, evaluation, checkpointing, and artifact writing.

Responsibilities:
- Parse PDBBind affinity labels.
- Build ligand atom graphs and protein pocket residue graphs.
- Define `ComplexRecord`, dataset, encoders, cross-attention, and `ProtLigGNN`.
- Train with MSE and Adam.
- Save best checkpoint, history CSV, test predictions, scatter plot, and training curves.

Used by:
- Direct CLI training.
- `infer_affinity.py` imports graph/model helpers.
- `score_generated_with_protliggnn.py` indirectly uses helpers through `infer_affinity.py`.
- `vina_validation.py` can score ligands using inference helpers.

### `infer_affinity.py`

Purpose:
- Single-pair inference for a protein PDB and ligand SDF/MOL2.

Responsibilities:
- Load checkpoint.
- Rebuild graph pair using training-time preprocessing.
- Run `ProtLigGNN` under `torch.no_grad`.
- Write `outputs/inference_result.json`.

### `genai_extract_ligands.py`

Purpose:
- Build a unique canonical SMILES corpus from local PDBBind ligand files.

Responsibilities:
- Walk local data for SDF/MOL2 files.
- Load molecules with RDKit.
- Convert to canonical SMILES.
- Deduplicate and save `outputs/genai/pdbbind_ligands_smiles.csv`.

### `train_smiles_generator.py`

Purpose:
- Train a character-level SMILES LSTM.

Responsibilities:
- Build character vocabulary with PAD/BOS/EOS.
- Create windowed training examples.
- Train next-character model with cross-entropy.
- Clip gradients at max norm 5.
- Save model, vocabulary, loss history, and PNG loss curve.

### `generate_and_filter_molecules.py`

Purpose:
- Sample molecules from the LSTM and filter them with RDKit.

Responsibilities:
- Load generator checkpoint and vocabulary.
- Sample SMILES with temperature.
- Canonicalize and deduplicate valid molecules.
- Compute QED, molecular weight, LogP, TPSA, HBD, HBA, Lipinski violations, novelty.
- Save generated molecule CSV and quality summary JSON.

### `score_generated_with_protliggnn.py`

Purpose:
- Score generated molecules against a target protein with ProtLigGNN.

Responsibilities:
- Convert generated SMILES into temporary 3D SDF with RDKit ETKDG and UFF.
- Use ProtLigGNN inference stack to predict affinity.
- Compute diversity and GAPS.
- Save ranked full list and top-10 CSVs.

### `vina_validation.py`

Purpose:
- Dock a single ligand or generated candidate batch with AutoDock Vina.

Responsibilities:
- Resolve Open Babel and Vina executables.
- Convert receptor and ligand files to PDBQT.
- Infer a blind docking box from receptor coordinates.
- Run Vina and parse scores.
- Save docking results, summaries, and simple PNG scatter plots.

### `dock_top_candidates.py`

Purpose:
- Dock known top-10 PDBBind predictions and generated top-10 GAPS candidates.

Responsibilities:
- Read top candidate CSVs.
- Locate known PDBBind complex files.
- Dock known ligands in their own receptor contexts.
- Dock generated ligands against hard-coded `2l3r` target protein.
- Save final docking comparison.

### `top_affinity_predictions.py`

Purpose:
- Aggregate test prediction CSVs and export top predicted known binders.

### `report_assets.py`

Purpose:
- Parse logs and output CSVs into report tables, plots, and a short text summary.

### `generate_paper_figures.py`

Purpose:
- Generate publication-style plots from existing histories, predictions, and comparison summaries.

### `final_research_package.py`

Purpose:
- Assemble compact final summaries from local output artifacts.

## 11. File Documentation

| File | Purpose | Inputs | Outputs | Key Dependencies |
|---|---|---|---|---|
| `protliggnn_train.py` | Affinity training and model implementation. | PDBBind index and complex files. | Checkpoint, CSVs, PNGs. | torch, PyG, RDKit, BioPython, SciPy, sklearn. |
| `benchmarking.py` | Shared benchmarking utilities. | Splits, predictions, metrics, run rows. | Split manifests, plots, summaries, manifests. | stdlib, optional matplotlib/RDKit/Torch introspection. |
| `experiment_infra.py` | Research infrastructure utilities. | Configs, dataset paths, split manifests, experiment dirs. | Hashes, registry rows, lockfiles, validations, lifecycle and reproducibility reports. | stdlib. |
| `benchmark.py` | Multi-seed benchmark runner. | Training CLI configuration and seeds. | Per-seed experiment runs and aggregate benchmark reports. | stdlib, `benchmarking.py`. |
| `compare_runs.py` | Compare saved experiment runs. | Experiment run IDs or paths. | Console table, comparison CSV, Markdown report. | stdlib, `benchmarking.py`, `experiment_infra.py`. |
| `infer_affinity.py` | Single-pair inference. | Protein PDB, ligand file, checkpoint. | `outputs/inference_result.json`. | torch, PyG, `protliggnn_train`. |
| `genai_extract_ligands.py` | Extract ligand SMILES. | Local PDBBind SDF/MOL2 files. | `pdbbind_ligands_smiles.csv`. | RDKit. |
| `train_smiles_generator.py` | Train SMILES LSTM. | Extracted SMILES CSV. | LSTM checkpoint, vocab, history, loss PNG. | torch. |
| `generate_and_filter_molecules.py` | Generate and filter molecules. | LSTM checkpoint and vocab. | Generated molecule CSV, quality JSON. | torch, RDKit. |
| `score_generated_with_protliggnn.py` | Score generated molecules. | Generated CSV, protein PDB, checkpoint. | Scored and top candidate CSVs. | torch, RDKit, PyG via inference. |
| `vina_validation.py` | Docking validation. | Protein PDB, ligand SMILES or generated CSV. | Docking CSV/TXT/PNG/JSON. | RDKit, Open Babel, AutoDock Vina. |
| `dock_top_candidates.py` | Dock top known/generated candidates. | Top known and generated CSVs. | Top-10 docking outputs and final comparison. | RDKit, Vina helpers. |
| `top_affinity_predictions.py` | Rank known predictions. | `test_predictions_*.csv`. | Top-10 CSV/TXT. | stdlib CSV. |
| `report_assets.py` | Build reporting assets. | Logs, histories, prediction CSVs. | Tables, plots, summary. | pandas, matplotlib. |
| `generate_paper_figures.py` | Build paper figures. | Existing outputs. | PNG figures. | matplotlib. |
| `final_research_package.py` | Compact final summaries. | Existing report, GenAI, docking artifacts. | Markdown/TXT/CSV package. | stdlib CSV/JSON/re. |
| `requirements.txt` | Dependency list. | None. | Install input. | Unpinned package names. |
| `.gitignore` | Version-control hygiene. | None. | Git ignore behavior. | Excludes data, checkpoints, large outputs. |

## 12. Class Documentation

### `ComplexRecord`
- Dataclass containing `pdb_id`, numeric affinity, protein path, and ligand path.
- Created by `discover_complex_records`.
- Consumed by `PDBbindPairDataset`.

### `PDBbindPairDataset`
- Eagerly converts `ComplexRecord` objects into in-memory samples.
- Each sample is `(ligand_graph, protein_graph, affinity, pdb_id)`.
- Maintains `skipped` records with error reasons.
- Design decision: eager processing simplifies training but hurts startup time and memory scalability.

### `LigandEncoder`
- Three-layer GAT encoder.
- Input: atom features `[num_atoms, 78]` and ligand edge index.
- Output: atom embeddings `[num_atoms, 256]`.
- Uses ELU and LayerNorm.

### `ProteinEncoder`
- Three-layer GCN encoder.
- Input: residue features `[num_residues, 30]` and protein residue edge index.
- Output: residue embeddings `[num_residues, 256]`.
- Uses ReLU and LayerNorm.

### `BidirectionalCrossAttention`
- Applies two multi-head attention modules:
  - ligand atoms query protein residues.
  - protein residues query ligand atoms.
- Adds residual connections and LayerNorm.
- Attention weights are not returned or stored.

### `ProtLigGNN`
- Full affinity regressor.
- Owns ligand encoder, protein encoder, cross-attention, dropout, and regressor MLP.
- Supports `no_crossgraph` ablation.
- Returns scalar predictions `[batch]`.

### `SmilesDataset`
- Encodes SMILES strings into token sequences with BOS/EOS.
- Crops random windows for long sequences.
- Used by SMILES generator training.

### `SmilesLSTMGenerator`
- Character-level LSTM generator in both training and generation scripts.
- Training version returns logits.
- Generation version also carries hidden state for autoregressive sampling.

## 13. Critical Function Documentation

| Function | File | Role | Notes |
|---|---|---|---|
| `parse_affinity_to_pkd` | `protliggnn_train.py` | Convert binding token to pAffinity. | Skips assay names with digits; no censor-aware handling. |
| `load_affinity_index` | `protliggnn_train.py` | Read PL index into PDB ID -> affinity map. | Uses only `INDEX_general_PL.2020R1.lst`. |
| `discover_complex_records` | `protliggnn_train.py` | Intersect local complex dirs with parsed labels. | Optional `max_samples` truncates sorted PDB IDs. |
| `load_ligand_molecule` | `protliggnn_train.py` | Load SDF/MOL2 with RDKit. | Requires conformer. |
| `atom_features` | `protliggnn_train.py` | Build 78-d atom vector. | Raises if feature size changes. |
| `mol_to_ligand_graph` | `protliggnn_train.py` | Convert RDKit Mol to PyG `Data`. | Stores `x`, `edge_index`, and `pos`. |
| `residue_features` | `protliggnn_train.py` | Build 30-d residue vector. | Standard amino acids only. |
| `build_protein_pocket_graph` | `protliggnn_train.py` | Build residue contact graph near ligand. | 6A pocket, 8A residue edges. |
| `process_complex` | `protliggnn_train.py` | Build one graph pair. | Robustness path returns error string. |
| `collate_pairs` | `protliggnn_train.py` | Build PyG batches. | Preserves PDB IDs. |
| `build_split_manifest` | `benchmarking.py` | Deterministic split manifest generation. | Supports random, fixed, scaffold, protein-family, and temporal strategies with warnings/fallbacks. |
| `run_epoch` | `protliggnn_train.py` | Train/eval one epoch. | MSE loss, optional optimizer. |
| `compute_metrics` | `protliggnn_train.py` | Delegate expanded regression metrics. | Uses shared metric implementation from `benchmarking.py`. |
| `save_checkpoint` | `protliggnn_train.py` | Persist model, optimizer, args, split, stats, and RNG state. | Backward-compatible with inference because `model_state_dict` and `args` remain present. |
| `build_graph_pair` | `infer_affinity.py` | Reuse training preprocessing for inference. | Single pair only. |
| `sample_smiles` | `generate_and_filter_molecules.py` | Autoregressive SMILES sampling. | Masks PAD and BOS. |
| `smiles_to_temp_sdf` | `score_generated_with_protliggnn.py` | Generate 3D ligand SDF. | ETKDG + UFF. |
| `add_gaps_scores` | `score_generated_with_protliggnn.py` | Compute GAPS rank score. | Fixed weights. |
| `docking_box_from_protein` | `vina_validation.py` | Blind-docking box from receptor extents. | Approximate. |
| `run_vina` | `vina_validation.py` | Execute Vina and parse score. | Uses command-list subprocess. |
| `parse_vina_score` | `vina_validation.py` | Extract Vina score from text. | Supports REMARK and table row formats. |

## 14. Configuration

Affinity training CLI:

| Argument | Default | Meaning |
|---|---:|---|
| `--data_dir` | `data/pdbbind2020` | Dataset root. |
| `--device` | `cpu` | Torch device string. |
| `--epochs` | `50` | Max training epochs. |
| `--batch_size` | `8` | Batch size in graph pairs. |
| `--lr` | `1e-3` | Adam learning rate. |
| `--weight_decay` | `1e-5` | Adam weight decay. |
| `--patience` | `10` | Early stopping patience. |
| `--seed` | `42` | Random seed. |
| `--max_samples` | `None` | Optional cap on discovered records. |
| `--run_name` | `protliggnn_run` | Output artifact naming. |
| `--no_crossgraph` | false | Disable cross-attention context. |
| `--save_path` | `outputs/best_protliggnn.pt` | Checkpoint path. |
| `--split_strategy` | `random` | Split mode: random, fixed, scaffold, protein_family, or temporal. |
| `--split_dir` | `splits` | Location for current split files. |
| `--fixed_split_dir` | `None` | Source directory for fixed train/val/test IDs. |
| `--val_fraction` | `0.1` | Validation fraction for generated splits. |
| `--test_fraction` | `0.1` | Test fraction for generated splits. |
| `--dataset_version` | `pdbbind2020_local` | Dataset version label written to artifacts and registry. |
| `--experiments_root` | `experiments` | Root for run directories and `registry.csv`. |
| `--experiment_dir` | `None` | Explicit experiment directory. |
| `--experiment_id` | `None` | Explicit experiment ID under `experiments_root`. |
| `--num_workers` | `0` | DataLoader worker count. |
| `--pin_memory` | false | DataLoader pin-memory flag. |
| `--disable_experiment_tracking` | false | Disable experiment directory/registry artifacts. |
| `--resume` | `None` | Resume target: `latest`, `run_XXX`, or checkpoint path. |
| `--allow_duplicate` | false | Allow rerunning identical config+dataset hash. |
| `--log_level` | `INFO` | Logging verbosity. |
| metadata fields | varies | Purpose, research question, author, description, notes, tags, expected outcome, milestone, architecture version, research version. |

Benchmark CLI:
- `python benchmark.py --seeds 5` runs seeds `42`, `123`, `777`, `2024`, and `3407`.
- `--allow_duplicate` is forwarded to training runs when intentional reruns are needed.
- `--dry_run` writes the benchmark plan without launching training.

Comparison CLI:
- `python compare_runs.py run_001 run_007` writes a console table, CSV, and Markdown comparison under `experiments/comparisons/`.

SMILES trainer CLI:
- `--epochs` default 30.
- `--batch_size` default 64.
- `--hidden_size` default 64.
- `--num_layers` default 1.
- `--lr` default `1e-3`.
- `--seq_len` default 96.

Generation CLI:
- `--num_samples` default 5000.
- `--max_length` default 120.
- `--temperature` default 0.8.

Generated scoring CLI:
- `--protein_pdb` required.
- `--checkpoint` default `outputs/best_protliggnn.pt`.
- `--device` default `cpu`.
- `--max_molecules` optional.

Vina CLI:
- `--protein_pdb` required.
- `--ligand_smiles` optional.
- `--generated_csv` default `outputs/genai/top10_generated_candidates_by_gaps.csv`.
- `--top_k` default 5.
- `--out_dir` default `outputs/vina_validation`.
- `--checkpoint` default `outputs/best_protliggnn.pt`.
- `--device` default `cpu`.

Environment variables:
- None are used directly by project code.

## 15. Dependencies

| Dependency | Why It Exists | Risk |
|---|---|---|
| `torch` | Neural network training, tensors, optimization. | Version unpinned; system Python currently lacks it. |
| `torch-geometric` | Graph data batches and GAT/GCN/global pooling. | Installation can be version/CUDA sensitive. |
| `rdkit-pypi` | Molecule loading, SMILES, descriptors, 3D embedding, fingerprints. | `rdkit-pypi` packaging can lag official RDKit. |
| `biopython` | PDB parsing and residue handling. | PDB quirks can affect graph construction. |
| `scipy` | Pearson/Spearman metrics. | Version unpinned. |
| `scikit-learn` | MAE/MSE metrics. | Version unpinned. |
| `pandas` | Report aggregation. | Version unpinned. |
| `numpy` | Geometry, arrays, random seed. | Version unpinned. |
| `tqdm` | Dataset processing progress. | Low risk. |
| `matplotlib` | Training and report plots. | Headless rendering can vary. |
| Open Babel | External molecule/receptor conversion to PDBQT. | Not a Python dependency; must be installed separately. |
| AutoDock Vina | Docking validation. | Not a Python dependency; executable path/platform dependent. |

## 16. Experiment Log

### Certified Optimized Baseline v1.1 (2026-07-12)
The repository baseline has been formally frozen and certified as **Optimized Baseline v1.1**. Metrics aggregated over 5 seeds (42, 123, 777, 2024, 3407) on the test set:
- **RMSE**: 1.605 ± 0.052 (95% CI: [1.533, 1.677])
- **MAE**: 1.268 ± 0.030
- **Pearson ($r$)**: 0.506 ± 0.048
- **Spearman ($\rho$)**: 0.491 ± 0.040
- **$R^2$**: 0.201 ± 0.079

### Multi-Seed Architecture Ablation Study (2026-07-12)
Controlled ablations evaluated across the same 5 seeds (42, 123, 777, 2024, 3407) on the test set:

| Configuration | Test RMSE | Test MAE | Test Pearson ($r$) | Test Spearman ($\rho$) | Test $R^2$ | Cohen's $d$ (vs. Baseline) | Paired $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Certified Baseline v1.1** | 1.605 ± 0.052 | 1.268 ± 0.030 | 0.506 ± 0.048 | 0.491 ± 0.040 | 0.201 ± 0.079 | — | — |
| **No Crossgraph** | 1.617 ± 0.061 | 1.277 ± 0.041 | 0.460 ± 0.061 | 0.437 ± 0.056 | 0.191 ± 0.072 | 0.202 | 0.580 |
| **No Attention** | 1.600 ± 0.055 | 1.267 ± 0.032 | 0.477 ± 0.051 | 0.452 ± 0.048 | 0.207 ± 0.071 | -0.078 | 0.816 |
| **No LayerNorm** | 1.673 ± 0.076 | 1.311 ± 0.065 | 0.392 ± 0.076 | 0.389 ± 0.072 | 0.135 ± 0.108 | 0.932 | 0.140 |
| **No Residual** | 1.588 ± 0.083 | 1.234 ± 0.051 | 0.503 ± 0.078 | 0.501 ± 0.071 | 0.221 ± 0.105 | -0.215 | 0.695 |
| **Max Pooling** | 1.556 ± 0.128 | 1.215 ± 0.097 | 0.560 ± 0.056 | 0.556 ± 0.057 | 0.255 ± 0.145 | -0.449 | 0.426 |

*Interpretation:* LayerNorm plays a massive role in stabilizing deep message-passing subgraphs (Cohen's $d = 0.932$). Cross-attention projections and skip connections are largely redundant under this pocket size. Max pooling exhibits high seed variance despite a lower mean RMSE.

### Retained Documented Legacy Experiments (Single-Seed)

| Run | Best Epoch | PCC | Spearman | RMSE | MAE | Notes |
|---|---:|---:|---:|---:|---:|---|
| `no_crossgraph_5000_e40` | 16 | 0.6714 | 0.6624 | 1.3722 | 1.0677 | Best retained run. |
| `crossgraph_5000_e40` | 21 | 0.6570 | 0.6495 | 1.3853 | 1.0642 | Crossgraph underperformed no-crossgraph on PCC/RMSE. |
| `crossgraph_attention_5000_e40` | 38 | 0.6023 | 0.5777 | 1.4905 | 1.1832 | Worse retained cross-attention variant. |
| `crossgraph_2000` | 12 | 0.4993 | 0.4408 | 1.5642 | 1.1890 | Earlier smaller run. |
| `no_crossgraph_2000` | 15 | 0.5110 | 0.4719 | 1.5912 | 1.2418 | Earlier smaller run. |
| `crossgraph_500` | 7 | 0.3554 | 0.3029 | 1.7626 | 1.3300 | Earlier smaller run. |
| `no_crossgraph_500` | 10 | 0.3392 | 0.2688 | 1.7778 | 1.2851 | Earlier smaller run. |

GenAI retained metrics:
- `generated_count`: 10000.
- `valid_count`: 6056.
- `unique_valid_count`: 4151.
- `novel_unique_count`: 3894.
- `validity_percentage`: 60.56.
- `uniqueness_percentage`: 68.5435931307794.
- `novelty_percentage`: 93.80872079017104.

Docking retained summary:
- Best generated Vina score: `-7.8`.
- Best generated SMILES: `NS(=O)(=O)c1cccc(-c2cc(C(F)(F)F)ccc2O)c1`.

Important caveat:
- The compact final summaries exist, but many underlying rich artifacts are currently absent from `outputs/`. Do not treat retained metrics as fully reproducible until checkpoints, logs, split manifests, and prediction CSVs are restored or regenerated.

## 17. Feature Changelog

| Date | Feature / Change | Files Changed | Reason | Architecture Impact | Research Impact | Compatibility |
|---|---|---|---|---|---|---|
| 2026-07-12 | Certified baseline and ran multi-seed ablations. | `run_multi_seed_ablations.py`, `ABLATION_STUDY.md`, `BRAIN.md` | Formalize baseline model, isolate and verify GNN sub-blocks across 5 seeds. | Frozen model config. | Confirmed LayerNorm is critical for pocket regression. | Full. |
| 2026-07-11 | Created implementation-first reverse-engineering report. | `PROTLIGGNN_REVERSE_ENGINEERING_REPORT.md` | Preserve architecture and research understanding from code. | Adds architectural reference. | Identifies supported and unsupported claims. | Additive. |
| 2026-07-11 | Created living engineering brain. | `BRAIN.md` | Mandated internal knowledge base for maintainability and onboarding. | Establishes documentation workflow. | Centralizes current research state and limitations. | Additive. |
| 2026-07-11 | Added reproducible benchmark and experiment framework. | `benchmarking.py`, `benchmark.py`, `protliggnn_train.py`, `BRAIN.md` | Make runs self-contained, multi-seed, statistically summarized, and split-reproducible. | Adds experiment-management layer without changing model architecture. | Enables publication-grade benchmark artifacts for new runs. | Backward-compatible legacy output files are still written. |
| 2026-07-11 | Added long-term research infrastructure hardening. | `experiment_infra.py`, `compare_runs.py`, `protliggnn_train.py`, `benchmark.py`, `BRAIN.md`, `VALIDATION_REPORT.md`, `ARCHITECTURE_DECISIONS.md` | Detect dataset/config drift, maintain a registry, support resume, validate artifacts, and compare runs. | Adds lifecycle infrastructure around the existing model without architecture changes. | Improves auditability, reproducibility, and publication-readiness of future experiments. | Additive CLI changes; duplicate reruns now require `--allow_duplicate` when an identical config+dataset hash is already registered. |

Historical implementation milestones inferred from current files:
- Core affinity model and training pipeline implemented in `protliggnn_train.py`.
- Single-pair inference implemented in `infer_affinity.py`.
- GenAI SMILES extraction/training/generation/scoring workflow implemented across four scripts.
- Vina validation implemented across `vina_validation.py` and `dock_top_candidates.py`.
- Reporting and final research package scripts implemented.

## 18. Design Decisions

| Decision | Why | Alternatives | Tradeoffs | Future Implication |
|---|---|---|---|---|
| Use PyG graph neural networks | Natural representation for molecules and residue contact graphs. | Dense grids, sequence models, descriptor-only ML. | Efficient sparse graphs but geometry is underused. | Add geometric GNNs or edge attributes. |
| Use hand-engineered atom/residue features | Simple, inspectable, dependency-light. | Pretrained embeddings, learned atom vocab, protein language models. | Easier to debug, less expressive. | Consider ESM/protein embeddings and richer atom/bond features. |
| Build pocket from full protein within 6A of ligand | Focus on local binding context. | Use provided pocket PDB, full protein, pocket prediction. | Depends on known ligand pose; may not fit apo/generated scenarios. | Need target-conditioned pocket definition for generation. |
| Use deterministic split manifests with multiple strategies | Publication experiments require exact train/val/test replay. | Ad hoc random split per run. | More artifact management, but much stronger reproducibility. | Historical random-split metrics remain non-reproducible unless regenerated. |
| Keep scripts instead of package | Fast prototype iteration. | Installable package with modules. | Low ceremony but harder to maintain/test. | Refactor into package before serious extension. |
| Add `--no_crossgraph` ablation preserving dimensions | Fair comparison while reusing head. | Separate head, remove module entirely. | Clean ablation, but zero-context concat is artificial. | Keep as baseline but add stronger baselines. |
| Use character-level SMILES LSTM | Lightweight and easy to train. | Transformer, graph generator, diffusion, fragment models. | Simple but weak and target-agnostic. | Upgrade generator for research contribution. |
| Use GAPS fixed weights | Transparent prioritization. | Learned ranker, Bayesian acquisition, Pareto ranking. | Easy to explain but arbitrary. | Validate or replace weights. |
| Use Vina for secondary validation | Independent physics-inspired check. | Docking ensembles, MD, experimental assays. | Useful screen but not ground truth. | Claims must remain cautious. |

## 19. Known Issues

High priority:
- Affinity parser skips `IC50` and other digit-containing assay names.
- Censored labels such as `<` are parsed as exact values.
- Historical retained experiments lack split manifests; new runs generate deterministic split manifests.
- Dependencies are unpinned.
- Local `venv` is broken/non-portable.
- No tests.
- Historical checkpoints may lack optimizer/RNG state; new checkpoints save optimizer and RNG state.
- Current retained outputs lack underlying rich artifacts for full reproducibility.

Medium priority:
- Cross-attention is not currently supported as a performance improvement by retained metrics.
- Protein graph ignores provided pocket PDB files.
- Coordinates are not used by the model after graph construction.
- Ligand bond types and protein edge distances are not modeled.
- DataLoader workers and pin memory are configurable, but optimal throughput settings are unbenchmarked.
- Per-complex Python cross-attention loop can bottleneck GPU usage.
- Generated molecule scoring is sequential and can be slow.

Research limitations:
- Random split likely overestimates generalization.
- No external baselines or literature benchmark comparison.
- No uncertainty estimates.
- Historical retained metrics lack confidence intervals and statistical tests; new multi-seed benchmarks generate aggregate statistics.
- Target-agnostic molecule generation.
- Docking box is blind/protein-extent based and approximate.

## 20. Future Roadmap

Immediate tasks:
1. Add tests for affinity parsing, graph construction, split behavior, model forward shape, and GAPS scoring.
2. Fix affinity parser to include `IC50`, `EC50`, and related assays.
3. Add tests for deterministic split manifests and fixed-split loading.
4. Pin dependencies in a lockfile or environment file.
5. Replace broken local environment assumptions with documented setup.
6. Add graph preprocessing cache (Done).

Medium-term goals:
1. Refactor scripts into an installable package, for example `protliggnn/`.
2. Validate scaffold, protein-family, and temporal split behavior on complete metadata.
3. Add ligand-only, protein-only, and descriptor baselines.
4. Add bond and distance edge attributes.
5. Add attention/feature attribution outputs.
6. Expand paired statistical testing once explicit baseline run groups are available.

Long-term research ideas:
1. Geometry-aware or equivariant protein-ligand interaction model.
2. Protein-conditioned molecular generation.
3. Active learning loop from generated/scored candidates.
4. Calibrated uncertainty and conformal screening.
5. Multi-objective optimization beyond fixed GAPS weights.
6. MD or experimental validation for top candidates.

Publication goals:
- Reproduce all retained metrics from clean environment.
- Establish rigorous splits and baselines.
- Report multiple seeds and confidence intervals.
- Make claims match implementation evidence.

## 21. Engineering Standards

Coding conventions:
- Prefer small, testable functions.
- Avoid hidden global state beyond constants.
- Keep CLI defaults documented here and in README if user-facing.
- Use structured parsers for scientific data where possible.
- Use clear names that map to domain concepts.

Repository conventions:
- Keep source files in version control.
- Keep large data, checkpoints, and local outputs ignored unless explicitly curated.
- Add generated research summaries only when they are compact and meaningful.
- If refactoring, preserve CLI compatibility or document breaking changes.

Testing requirements:
- Every bug fix should add or update a test when feasible.
- Graph construction changes require fixture-based tests.
- Model architecture changes require shape tests.
- Parser changes require representative label-token tests.
- Reporting changes require minimal CSV/text fixture tests.

Documentation standards:
- Update `BRAIN.md` after every meaningful code, experiment, or architecture change.
- Keep `PROTLIGGNN_REVERSE_ENGINEERING_REPORT.md` as implementation-derived reference.
- Do not overclaim beyond available artifacts.
- Keep historical experiment records.

Commit philosophy:
- Small commits grouped by purpose.
- Separate code changes from generated artifacts where possible.
- Include architecture impact in commit messages for large changes.

## 22. Reverse Engineering Summary

Important findings from `PROTLIGGNN_REVERSE_ENGINEERING_REPORT.md`:

- The implementation, not the README, shows this is a protein-ligand affinity regressor plus generated-ligand prioritization workflow.
- Core model: ligand GAT, protein GCN, optional bidirectional cross-attention, mean pooling, MLP regression.
- The label parser only supports alphabetic assay names and currently skips many local complexes.
- The local data package is v2020 entries with PDBbind v2024 reprocessed structure files according to `data/pdbbind2020/index/README`.
- The retained best model is no-crossgraph, not cross-attention.
- The GenAI component is a lightweight character-level SMILES LSTM.
- Vina is secondary validation, not ground truth.
- Current project state is research prototype, not publication-ready.
- Highest-impact improvements are parser correctness, split rigor, dependency pinning, tests, graph caching, geometry-aware modeling, and artifact manifests.

## 23. Developer Onboarding Guide

Read first:
1. `BRAIN.md`.
2. `PROTLIGGNN_REVERSE_ENGINEERING_REPORT.md`.
3. `protliggnn_train.py`.
4. `infer_affinity.py`.
5. GenAI and docking scripts only if working on those workflows.

Expected setup:
Refer to [OFFICIAL_ENVIRONMENT.md](file:///d:/ProtLigGnn/OFFICIAL_ENVIRONMENT.md) for step-by-step instructions. The official environment requires Python 3.11.9, PyTorch 2.3.1+cu121, CUDA 12.1, and PyG 2.8.0.

Optional external tools:
  Open Babel
  AutoDock Vina

Train affinity model:
```bash
python protliggnn_train.py --data_dir data/pdbbind2020 --device cpu --epochs 40 --max_samples 5000 --run_name my_run
```

Train no-crossgraph baseline:
```bash
python protliggnn_train.py --data_dir data/pdbbind2020 --device cpu --epochs 40 --max_samples 5000 --no_crossgraph --run_name no_crossgraph_5000_e40
```

Run single inference:
```bash
python infer_affinity.py --protein_pdb path/to/protein.pdb --ligand_file path/to/ligand.sdf --checkpoint outputs/best_protliggnn.pt --device cpu
```

Run GenAI flow:
```bash
python genai_extract_ligands.py
python train_smiles_generator.py --epochs 30 --batch_size 64
python generate_and_filter_molecules.py --num_samples 5000 --temperature 0.8
python score_generated_with_protliggnn.py --protein_pdb data/pdbbind2020/pdbbind_subset/2011-2019/2l3r/2l3r_protein.pdb --checkpoint outputs/best_protliggnn.pt --device cpu
```

Run Vina validation:
```bash
python vina_validation.py --protein_pdb data/pdbbind2020/pdbbind_subset/2011-2019/2l3r/2l3r_protein.pdb --generated_csv outputs/genai/top10_generated_candidates_by_gaps.csv
```

Common debugging:
- If no records are found, check data root and index path.
- If many records are skipped, inspect RDKit loading errors and affinity parser coverage.
- If inference fails, verify checkpoint compatibility and dependency installation.
- If docking fails, verify Open Babel and Vina executables.
- If generated scoring skips molecules, inspect RDKit 3D embedding failures.

## 24. Architecture Diagrams

Repository dependency graph:

```text
protliggnn_train.py
    ^
    |
infer_affinity.py
    ^
    |
score_generated_with_protliggnn.py

vina_validation.py <---- dock_top_candidates.py

genai_extract_ligands.py
    -> train_smiles_generator.py
        -> generate_and_filter_molecules.py
            -> score_generated_with_protliggnn.py

report_assets.py
generate_paper_figures.py
top_affinity_predictions.py
    -> final_research_package.py
```

Training sequence:

```text
CLI args
  -> set_seed
  -> discover_complex_records
  -> PDBbindPairDataset
  -> build_split_manifest
  -> DataLoader
  -> ProtLigGNN
  -> Adam + MSE
  -> validation metrics
  -> checkpoint
  -> test metrics
  -> artifacts
```

Inference sequence:

```text
protein_pdb + ligand_file + checkpoint
  -> load_model
  -> load_ligand_molecule
  -> mol_to_ligand_graph
  -> build_protein_pocket_graph
  -> Batch.from_data_list
  -> model.forward
  -> predicted_affinity
```

Model architecture:

```text
Ligand atom graph [N_l, 78]     Protein pocket graph [N_p, 30]
          |                                  |
          v                                  v
    LigandEncoder                       ProteinEncoder
    GAT x 3 -> [N_l,256]                GCN x 3 -> [N_p,256]
          |                                  |
          +---------------+------------------+
                          v
          BidirectionalCrossAttention or zero context
                          |
                          v
      ligand [N_l,512]              protein [N_p,512]
               |                         |
               v                         v
        global_mean_pool          global_mean_pool
               |                         |
               +-----------+-------------+
                           v
                    concat [B,1024]
                           |
                           v
                   MLP -> affinity [B]
```

Class hierarchy:

```text
torch.nn.Module
|-- LigandEncoder
|-- ProteinEncoder
|-- BidirectionalCrossAttention
|-- ProtLigGNN
`-- SmilesLSTMGenerator

torch.utils.data.Dataset
|-- PDBbindPairDataset
`-- SmilesDataset
```

## 25. Current Project State

Current milestone:
- Research infrastructure hardening implemented for new runs.

Implemented:
- Affinity training.
- Single-pair inference.
- Ligand SMILES extraction.
- SMILES LSTM training.
- Molecule generation/filtering.
- ProtLigGNN scoring for generated molecules.
- GAPS ranking.
- Vina validation.
- Report/final-summary generation.
- Reverse-engineering report.
- Living `BRAIN.md`.
- Deterministic split manifests.
- Self-contained experiment directories.
- Multi-seed benchmark runner.
- Expanded regression metrics, plots, error analysis, and aggregate statistics.
- Dataset fingerprinting.
- Configuration hashing and duplicate detection.
- Experiment registry.
- Structured experiment logging.
- Environment freeze artifacts.
- Split validation.
- Experiment metadata.
- Artifact validation.
- Reproducibility scoring.
- Resume target resolution and checkpoint restoration.
- Experiment comparison utility.
- Architecture decision records.

Pending:
- Tests (Done).
- Dependency pinning (Done).
- Graph cache (Done).
- Rigorous baselines.
- Geometry-aware modeling.
- Full end-to-end rerun in a valid Torch/PyG environment (Done).

Current best retained performance:
- `no_crossgraph_5000_e40`.
- PCC 0.6714.
- Spearman 0.6624.
- RMSE 1.3722.
- MAE 1.0677.

Current best checkpoint:
- Intended default path is `outputs/best_protliggnn.pt`.
- Current visible output tree does not show this checkpoint. Verify before running inference.

Known blockers:
- Local `venv` is broken.
- Available Python installations in the latest verification had no `torch`.
- Rich experiment artifacts are missing/deleted from current visible outputs.
- Parser coverage issue reduces usable local data.
- Full resume/training/inference execution still requires a valid Torch/PyG environment.

Next priorities:
1. Create reproducible environment.
2. Add parser tests and fix parser.
3. Add graph/model smoke tests.
4. Run full benchmark from a clean Torch/PyG environment.
5. Reproduce baseline metrics from scratch with new artifacts.
6. Validate resume continuation numerically once Torch/PyG is available.

## 26. Reproducible Benchmarking Milestone

Status:
- Implemented on 2026-07-11.
- Scope was experimental methodology, benchmarking, reproducibility, artifact management, and evaluation quality.
- The GNN architecture, dataset definition, target labels, and research hypothesis were not redesigned.

New files:
- `benchmarking.py`: deterministic split generation, experiment artifact utilities, environment capture, regression metrics, plots, error analysis, statistical aggregation, and manifest writing.
- `benchmark.py`: one-command multi-seed benchmark runner that launches `protliggnn_train.py` once per seed and aggregates completed run metrics.
- `experiment_infra.py`: dataset/config fingerprinting, registry, resume resolution, environment freezing, split/artifact validation, lifecycle docs, logging setup, and reproducibility scoring.
- `compare_runs.py`: artifact-based comparison of saved experiment runs.
- `VALIDATION_REPORT.md`: milestone validation report and known execution blockers.

Updated file:
- `protliggnn_train.py`: now creates experiment directories by default while retaining legacy `outputs/` history, prediction, scatter, training-curve, and checkpoint behavior.

New repository workflow:

```text
python benchmark.py --seeds 5 --epochs 50 --max_samples 5000 --no_crossgraph
  |
  |-- experiments/run_001/
  |     |-- config.json
  |     |-- cli_args.json
  |     |-- environment.json
  |     |-- environment.lock
  |     |-- requirements.lock
  |     |-- pip_freeze.txt
  |     |-- git.json
  |     |-- metadata.json
  |     |-- config_hash.txt
  |     |-- config_fingerprint.json
  |     |-- dataset_hash.json
  |     |-- dataset_metadata.json
  |     |-- split/
  |     |   |-- train_ids.txt
  |     |   |-- val_ids.txt
  |     |   |-- test_ids.txt
  |     |   `-- metadata.json
  |     |-- split_validation.json
  |     |-- metrics.json
  |     |-- predictions.csv
  |     |-- error_analysis.csv
  |     |-- error_analysis.md
  |     |-- evaluation_report.md
  |     |-- training_history.csv
  |     |-- checkpoint.pt
  |     |-- plots/
  |     |-- experiment.log
  |     |-- artifact_validation.json
  |     |-- reproducibility_report.md
  |     |-- reproducibility_report.json
  |     |-- experiment_lifecycle.md
  |     `-- manifest.json
  |
  |-- experiments/run_002/
  |-- ...
  `-- experiments/benchmark_<timestamp>/
        |-- benchmark_plan.json
        |-- completed_runs.json
        |-- per_seed_metrics.csv
        |-- results_summary.csv
        |-- benchmark_summary.json
        `-- benchmark_report.md
```

`experiments/registry.csv` is the master experiment index.

Split system:
- Default strategy is deterministic seed split over processed PDB IDs.
- `--split_strategy fixed` loads `train_ids.txt`, `val_ids.txt`, and `test_ids.txt` from `--fixed_split_dir` or `--split_dir`.
- `--split_strategy scaffold` groups complexes by RDKit Murcko scaffold where ligand parsing permits; missing scaffold extraction falls back to singleton groups with warnings.
- `--split_strategy protein_family` uses explicit family metadata when present; otherwise it falls back to deterministic PDB-prefix groups with warnings.
- `--split_strategy temporal` uses PDBbind release-year metadata from `INDEX_general_PL.2020R1.lst` when complete; otherwise it falls back to deterministic seed split with warnings.
- Every split writes `splits/train_ids.txt`, `splits/val_ids.txt`, `splits/test_ids.txt`, and `splits/metadata.json`, then copies the exact manifest into the experiment directory.

Evaluation package:
- Test metrics now include MSE, RMSE, MAE, median absolute error, Pearson, Spearman, R2, bias, and mean prediction.
- Plots include prediction scatter, residual scatter, error histogram, residual distribution, prediction histogram, calibration curve, and calibration CSV.
- Error analysis stores all prediction residuals, top largest errors, best predictions, outliers, bias, and average prediction.

Statistical aggregation:
- `benchmark.py --seeds 5` expands to seeds `42`, `123`, `777`, `2024`, and `3407`.
- Aggregation writes mean, median, standard deviation, min, max, normal-theory 95 percent CI, and bootstrap 95 percent CI.
- Paired statistical tests are explicitly marked not applicable until baseline run groups are supplied.

Research-infrastructure additions after the initial reproducibility milestone:
- Every tracked experiment writes `dataset_hash.json` with SHA256, file size, and modified-time records for dataset index files, label files, metadata, and split manifests.
- Every tracked experiment writes `config_hash.txt` and `config_fingerprint.json` from a canonical, output-path-stable configuration.
- `experiments/registry.csv` records run ID, timestamp, git commit, dataset/config hash, split strategy, seed, best validation RMSE, test metrics, status, duration, and checkpoint path.
- Duplicate config+dataset hashes are detected before training and blocked unless `--allow_duplicate` is supplied.
- `--resume latest`, `--resume run_004`, and `--resume <checkpoint>` restore model, optimizer, scheduler if present, epoch, and RNG state when those fields exist in the checkpoint.
- `experiment.log` uses Python logging with timestamped INFO/WARNING/ERROR/DEBUG events.
- Environment freeze artifacts are written as `environment.lock`, `requirements.lock`, and `pip_freeze.txt`.
- `split_validation.json` fails the run on train/validation/test overlap, duplicate IDs, missing IDs, or orphan IDs.
- `metadata.json` stores searchable experiment purpose, research question, author, description, notes, tags, expected outcome, milestone, architecture version, and research version.
- `artifact_validation.json` verifies required outputs before the run is marked successful.
- `reproducibility_report.md` assigns a 0-100 reproducibility score and recommendations.
- `experiment_lifecycle.md` documents the CLI-to-registry lifecycle for the run.
- `compare_runs.py run_001 run_007` generates a console table, CSV, and Markdown comparison.

Backward compatibility:
- Existing `protliggnn_train.py` CLI arguments remain valid.
- Legacy files under `outputs/` are still written.
- New arguments are additive: `--split_strategy`, `--split_dir`, `--fixed_split_dir`, `--val_fraction`, `--test_fraction`, `--dataset_version`, `--experiments_root`, `--experiment_dir`, `--experiment_id`, `--num_workers`, `--pin_memory`, `--disable_experiment_tracking`, `--resume`, `--allow_duplicate`, `--log_level`, and experiment metadata fields.

Reverse-engineering summary update:
- The implementation still supports the same research objective: graph-based protein-ligand binding affinity regression on PDBbind-style complexes.
- The main methodological weakness identified in the reverse-engineering report was not the GNN architecture but experiment rigor: random splits, weak artifact traceability, narrow metrics, and no multi-seed statistics.
- This milestone addresses those weaknesses for new experiments without claiming improved accuracy or changing the underlying scientific hypothesis.

## 27. Phase 3.1A — Geometry Feature Injection (Ligand-Only)

Status:
- Completed benchmarking and auditing on 2026-07-12.
- Outcome: Certified Equivalent (Baseline v1.1 retained).
- Detailed metrics: Mean RMSE = 1.5442 +/- 0.0818 (paired t-test p = 0.1484), Mean MAE = 1.2139 +/- 0.0628 (p = 0.0564), Mean PCC = 0.5470 +/- 0.0197 (p = 0.2092).
- The scope was strictly controlled: explicit 3D Euclidean distances were injected into the ligand GATConv layers only, while the protein encoder and all other modules remained 100% identical to Baseline v1.1.

New files:
- `models/geometry/radial_basis.py`: Gaussian Radial Basis Function (RBF) expansion mapping distances.
- `models/geometry/distance_encoding.py`: DistanceEncoder projecting RBFs to 32 dimensions.
- `models/geometry/geometry_utils.py`: Dataclass `GeometryFeatures` and distance helper functions.
- `models/geometry/geometry_model.py`: Upgraded `GeometryAwareLigandEncoder` and model wrapper `ProtLigGNNGeometry`.
- `tests/test_geometry.py`: Unit tests for RBF, shapes, backpropagation, and determinism.
- `tests/test_baseline_equivalence.py`: Protection tests verifying original baseline logic and parameters.
- `tests/test_benchmark_equivalence.py`: Protection tests verifying graph cache checksums and split IDs.

Updated files:
- `protliggnn_train.py`: Modified to import the geometry model and add the `--model_type` CLI option.

Workflow:
```bash
python protliggnn_train.py --model_type geometry --epochs 40 --device cuda --run_name my_geom_run
```
All unit tests and baseline/benchmark protection tests pass successfully.

