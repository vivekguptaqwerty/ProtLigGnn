# ProtLigGNN Implementation-First Reverse Engineering Report

Date: 2026-07-11
Workspace: `D:\ProtLigGnn`

Method: I treated source code, training scripts, data loaders, model code, generated artifacts, and dependency structure as the primary evidence. README and generated prose summaries were read only after the implementation map was complete.

Legend:
- Observed Fact: directly visible in implementation, local data, dependency files, or generated artifacts.
- Strong Inference: the most likely research intent given repeated implementation choices.
- Hypothesis: plausible but not fully established by the current code/artifacts.

## 1. Executive Summary

- Observed Fact: The repository implements a Python/PyTorch Geometric protein-ligand binding affinity regressor named `ProtLigGNN`, plus a small SMILES-generation and docking-validation workflow. Primary files: `protliggnn_train.py`, `infer_affinity.py`, `train_smiles_generator.py`, `generate_and_filter_molecules.py`, `score_generated_with_protliggnn.py`, `vina_validation.py`, and `dock_top_candidates.py`.
- Observed Fact: The core affinity model converts ligands into atom graphs, converts proteins into ligand-centered pocket residue graphs, encodes ligand atoms with three `GATConv` layers, encodes protein residues with three `GCNConv` layers, optionally applies bidirectional cross-attention, mean-pools both graphs, and regresses a scalar affinity using MSE loss (`protliggnn_train.py:27`, `protliggnn_train.py:541`, `protliggnn_train.py:558`, `protliggnn_train.py:575`, `protliggnn_train.py:621`, `protliggnn_train.py:728`).
- Strong Inference: The main scientific question is whether paired ligand and protein pocket graph representations can predict PDBBind affinity labels well enough to prioritize known and generated binders.
- Observed Fact: The documented "best" retained result says the no-crossgraph ablation outperformed the cross-attention versions: `no_crossgraph_5000_e40` PCC 0.6714, Spearman 0.6624, RMSE 1.3722, MAE 1.0677 (`outputs/final_research_package/final_results_summary.md:13`, `outputs/final_research_package/final_results_summary.md:23`).
- Strong Inference: The current implementation is closer to a prototype research pipeline than a publishable benchmark system: it lacks deterministic split manifests, cached graph preprocessing, environment lockfiles, automated tests, uncertainty estimates, statistical significance, robust external baselines, and scalable data loading.

## 2. Repository Discovery

Observed Fact: Top-level first-party source is compact: 12 Python scripts, `requirements.txt`, `.gitignore`, `README.md`, local `data/`, local `outputs/`, local `venv/`, and `__pycache__/`.

Structural tree, collapsed for large repeated data:

```text
D:\ProtLigGnn
|-- .git/
|-- .gitignore
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
|       |   |-- README
|       |   |-- INDEX_general_PL.2020R1.lst
|       |   |-- INDEX_general_PN.2020R1.lst
|       |   |-- INDEX_general_PP.2020R1.lst
|       |   `-- INDEX_general_NL.2020R1.lst
|       `-- pdbbind_subset/
|           `-- 2011-2019/
|               `-- <12235 complex dirs>/
|                   |-- <pdb_id>_protein.pdb
|                   |-- <pdb_id>_pocket.pdb
|                   |-- <pdb_id>_ligand.sdf
|                   `-- <pdb_id>_ligand.mol2
|-- outputs/
|   `-- final_research_package/
|       |-- final_results_summary.md
|       |-- final_results_summary.txt
|       |-- key_metrics_table.csv
|       |-- key_metrics_table.md
|       |-- limitations_and_future_work.md
|       `-- top_generated_candidate_summary.md
|-- venv/
`-- __pycache__/
```

Folder purposes:
- Observed Fact: `data/pdbbind2020/index` stores PDBBind index files and metadata. `INDEX_general_PL.2020R1.lst` is the only affinity index used by training (`protliggnn_train.py:29`, `protliggnn_train.py:181`).
- Observed Fact: `data/pdbbind2020/pdbbind_subset/2011-2019` stores local PDBBind complex folders. There are 12,235 complex directories, each with the expected protein, pocket, SDF, and MOL2 files.
- Observed Fact: `outputs/final_research_package` stores compact generated summaries. Most richer experiment outputs are currently absent or deleted in the worktree.
- Observed Fact: `venv` is present, but `venv/pyvenv.cfg` points to `C:\Users\Admin\AppData\Local\Programs\Python\Python310`, making it non-portable on this machine.

Technology stack:
- Observed Fact: Python only. Dependencies are `torch`, `torch-geometric`, `rdkit-pypi`, `biopython`, `scipy`, `scikit-learn`, `pandas`, `numpy`, `tqdm`, and `matplotlib` (`requirements.txt`).
- Observed Fact: Configuration is `argparse`, not Hydra/OmegaConf/YAML (`protliggnn_train.py:867`, `train_smiles_generator.py:69`, `vina_validation.py:22`).
- Observed Fact: Experiment tracking is CSV, PNG, text logs, and checkpoint files. No W&B, MLflow, TensorBoard, or database registry was found.
- Observed Fact: CUDA is only exposed through `--device` and CUDA seed calls if available (`protliggnn_train.py:138`, `protliggnn_train.py:870`). There is no AMP, multi-GPU, DDP, scheduler, torch.compile, `num_workers`, `pin_memory`, or persistent workers.

## 3. Reverse-Engineered Research Objective

This project appears to investigate protein-ligand binding affinity prediction and affinity-guided ligand prioritization because the implementation demonstrates:

- Observed Fact: PDBBind protein-ligand complexes are discovered from local structure files and labels (`protliggnn_train.py:181`, `protliggnn_train.py:202`, `protliggnn_train.py:231`).
- Observed Fact: Labels are converted into `-log10(molar affinity)`, a pKd/pKi-like continuous target (`protliggnn_train.py:160`).
- Observed Fact: The model predicts a scalar affinity with MSE loss and reports PCC, Spearman, RMSE, and MAE (`protliggnn_train.py:728`, `protliggnn_train.py:754`).
- Observed Fact: Generated molecules are ranked by predicted affinity and by GAPS, then top molecules are docked with AutoDock Vina (`score_generated_with_protliggnn.py:146`, `dock_top_candidates.py:138`, `vina_validation.py:147`).

Problem statement:
- Strong Inference: Given a ligand 3D structure and a protein receptor structure, predict binding strength on a continuous pAffinity scale and use that predictor for ranking both known PDBBind ligands and generated molecules.

Scientific hypothesis:
- Hypothesis: A paired ligand-atom graph plus protein-pocket residue graph can learn chemically meaningful interaction patterns from PDBBind, and the learned predictor can serve as a scoring function inside a lightweight generative chemistry loop.

Expected contribution:
- Strong Inference: The intended contribution is not a novel graph convolution, RDKit descriptor, or docking algorithm. It is an integrated closed-loop pipeline: PDBBind training -> graph affinity model -> SMILES generation -> RDKit filtering -> affinity/GAPS prioritization -> Vina validation.

## 4. End-to-End Research Pipeline

```text
PDBBind index + complex dirs
    |
    v
label parsing: Kd/Ki token -> pAffinity = -log10(M)
    |
    v
ligand SDF/MOL2 -> RDKit Mol -> atom graph
protein PDB + ligand coords -> residues within 6A -> pocket graph
    |
    v
PDBbindPairDataset eagerly materializes graph pairs
    |
    v
random 80/10/10 split by seed
    |
    v
Ligand GAT encoder + Protein GCN encoder
    |
    v
optional bidirectional cross-attention per complex
    |
    v
global mean pooling -> MLP regression head
    |
    v
MSE training, Adam optimizer, early stopping by validation RMSE
    |
    v
test metrics + checkpoint + CSV/PNG artifacts
    |
    v
inference and generated-molecule scoring
    |
    v
GAPS ranking + AutoDock Vina secondary validation
```

## 5. Dataset Analysis

- Observed Fact: The implementation reads `data/pdbbind2020/index/INDEX_general_PL.2020R1.lst` (`protliggnn_train.py:29`, `protliggnn_train.py:181`).
- Observed Fact: The local index contains 19,037 protein-ligand entries. The local subset contains 12,235 complex directories. Every local complex dir has expected protein, pocket, SDF, and MOL2 files.
- Observed Fact: The exact parser regex accepts only all-letter measurement names: `([A-Za-z]+)([<>=]+)(number)(unit)` (`protliggnn_train.py:160`). In this local subset, that yields 7,335 parseable complex labels. It skips 4,900 local complexes, mainly because common labels containing digits such as `IC50` do not match the regex.
- Observed Fact: Parsed subset label types are Kd 4,836 and Ki 2,499; relation counts are 7,270 exact/equality-style and 65 less-than. Units are uM 4,093, nM 2,812, mM 332, and pM 98.
- Observed Fact: Parsed local pAffinity range is approximately 0.3979 to 12.6990 with mean 6.2041.
- Observed Fact: Ligands prefer SDF over MOL2 (`protliggnn_train.py:214`), are loaded with RDKit without removing hydrogens, and require a 3D conformer (`protliggnn_train.py:257`).
- Observed Fact: The protein graph is not the provided `<pdb_id>_pocket.pdb`; it is rebuilt from `<pdb_id>_protein.pdb` by selecting residues whose atoms lie within 6A of ligand atoms (`protliggnn_train.py:429`, `protliggnn_train.py:448`).
- Strong Inference: The 6A residue filter is intended to focus learning on the binding pocket and reduce graph size/noise.
- Observed Fact: Splitting is random by sample index with 80/10/10 proportions for sample counts >=10 (`protliggnn_train.py:677`). There is no scaffold split, target-family split, time split, or saved split manifest.
- Reproducibility Risk: Random splits over PDB IDs can leak homologous proteins or similar ligands across train/test and inflate performance relative to stricter benchmark protocols.

## 6. Feature Engineering

Ligand node features:
- Observed Fact: Ligand feature dimension is fixed at 78 (`protliggnn_train.py:27`).
- Observed Fact: Features include atom number, degree, formal charge, hybridization, hydrogen count, implicit/total valence, chirality, ring sizes 3-8, aromaticity, ring membership, isotope, mass, van der Waals/covalent radii, radical electrons, chirality flags, donor/acceptor/acidic/basic flags, halogen and metal flags, and normalized degree (`protliggnn_train.py:298`, `protliggnn_train.py:321`, `protliggnn_train.py:332`).
- Strong Inference: The ligand encoder is hand-designed to combine chemical identity, topology, physicochemical hints, and stereochemical signals.

Protein residue node features:
- Observed Fact: Protein feature dimension is fixed at 30 (`protliggnn_train.py:28`).
- Observed Fact: Features include 20 amino acid one-hot dimensions, aromatic/hydrophobic/polar/positive/negative group flags, glycine/proline flags, CA-present flag, normalized residue mass, and side-chain atom count proxy (`protliggnn_train.py:384`, `protliggnn_train.py:394`).
- Limitation: Coordinates are stored in `Data.pos` but not directly consumed by the GCN/GAT layers except through graph construction. Distance values are not edge attributes.

Graph edges:
- Observed Fact: Ligand edges are directed versions of RDKit covalent bonds (`protliggnn_train.py:355`, `protliggnn_train.py:365`).
- Observed Fact: Protein residue edges connect residue centers within 8A, with a sequential fallback if no edges exist (`protliggnn_train.py:461`, `protliggnn_train.py:465`, `protliggnn_train.py:468`).
- Limitation: Bond type, inter-residue distance, direction, and 3D geometry are not used as edge attributes.

## 7. Model Architecture

```text
Ligand graph                         Protein pocket graph
x_l: [N_l, 78]                       x_p: [N_p, 30]
edge_l: covalent bonds               edge_p: residue-center <= 8A
    |                                      |
    v                                      v
GATConv 78 -> 256                    GCNConv 30 -> 256
LayerNorm + ELU                      LayerNorm + ReLU
GATConv 256 -> 256                   GCNConv 256 -> 256
LayerNorm + ELU                      LayerNorm + ReLU
GATConv 256 -> 256                   GCNConv 256 -> 256
LayerNorm + ELU                      LayerNorm + ReLU
    |                                      |
    `-------- per-complex cross attention -'
              ligand attends protein
              protein attends ligand
              residual + LayerNorm
                     |
     concat original/context -> 512 each side
                     |
        global_mean_pool ligand and protein
                     |
         concat pools -> [B, 1024]
                     |
      Dropout -> Linear 1024->512 -> ReLU
               -> Dropout -> Linear 512->128
               -> ReLU -> Dropout -> Linear 128->1
```

Observed architecture evidence:
- `LigandEncoder` uses three GAT layers with hidden dimension 256 (`protliggnn_train.py:541`, `protliggnn_train.py:544`).
- `ProteinEncoder` uses three GCN layers with hidden dimension 256 (`protliggnn_train.py:558`, `protliggnn_train.py:561`).
- `BidirectionalCrossAttention` uses PyTorch `nn.MultiheadAttention` with 4 heads and dropout 0.1 (`protliggnn_train.py:575`, `protliggnn_train.py:579`, `protliggnn_train.py:586`).
- `ProtLigGNN` concatenates original and cross-context features, mean-pools, then uses a 1024-input MLP (`protliggnn_train.py:641`, `protliggnn_train.py:671`, `protliggnn_train.py:631`).
- Observed Fact: `--no_crossgraph` replaces cross-context with zeros while keeping the same downstream tensor size, enabling a controlled architectural ablation (`protliggnn_train.py:641`, `protliggnn_train.py:879`).

## 8. Layer-by-Layer Analysis

| Layer | Input | Output | Operation | Purpose | Limitation |
|---|---:|---:|---|---|---|
| Ligand GAT1 | `[N_l, 78]` | `[N_l, 256]` | 4-head graph attention, 64 per head, concat | Learn atom representations weighted by bonded neighbors | Only covalent graph, no bond features |
| Ligand GAT2 | `[N_l, 256]` | `[N_l, 256]` | 4-head GAT | Higher-order molecular context | No explicit distance/3D geometry |
| Ligand GAT3 | `[N_l, 256]` | `[N_l, 256]` | single-head GAT concat false | Final atom embedding | Attention weights not exposed |
| Protein GCN1 | `[N_p, 30]` | `[N_p, 256]` | normalized graph convolution | Smooth residue features over pocket contact graph | Residue graph is coarse |
| Protein GCN2 | `[N_p, 256]` | `[N_p, 256]` | GCN | Expand pocket context | Edge distance ignored |
| Protein GCN3 | `[N_p, 256]` | `[N_p, 256]` | GCN | Final residue embedding | No sequence model or pretrained protein LM |
| Cross attention L->P | `[N_l, 256]`, `[N_p, 256]` | `[N_l, 256]` | atom queries, residue keys/values | Inject pocket context into ligand atoms | O(N_l*N_p), no attention mask beyond complex split |
| Cross attention P->L | `[N_p, 256]`, `[N_l, 256]` | `[N_p, 256]` | residue queries, atom keys/values | Inject ligand context into residues | Attention maps discarded (`need_weights=False`) |
| Concatenation | `[N,256]+[N,256]` | `[N,512]` | feature concat | Preserve original and interaction-conditioned features | Doubles feature size |
| Global mean pool | node embeddings | `[B,512]` each | average over graph nodes | Fixed-size graph representation | Mean pooling can dilute rare interaction motifs |
| Regressor | `[B,1024]` | `[B]` | MLP 1024-512-128-1 | Scalar affinity regression | No uncertainty or calibration |

## 9. Forward Pass Walkthrough

Symbolic batch with B complexes:

```text
ligand_batch.x:  [sum_i N_li, 78]
protein_batch.x: [sum_i N_pi, 30]

ligand_encoder -> [sum_i N_li, 256]
protein_encoder -> [sum_i N_pi, 256]

for each complex i:
  ligand_i:  [N_li, 256]
  protein_i: [N_pi, 256]
  attention:
    Q_l = ligand_i, K_p/V_p = protein_i -> ligand_context_i [N_li, 256]
    Q_p = protein_i, K_l/V_l = ligand_i -> protein_context_i [N_pi, 256]
  residual + norm
  concat:
    ligand_joint_i  [N_li, 512]
    protein_joint_i [N_pi, 512]

cat all complexes:
  ligand_joint:  [sum_i N_li, 512]
  protein_joint: [sum_i N_pi, 512]

global_mean_pool:
  ligand_pool:  [B, 512]
  protein_pool: [B, 512]
  joint:        [B, 1024]

MLP:
  [B,1024] -> [B,512] -> [B,128] -> [B,1] -> [B]
```

Observed Fact: I attempted a local executable shape dry run, but the checked-in `venv` points to a missing Python 3.10 path and system Python 3.14 lacks `torch`. Therefore, the tensor walkthrough is derived from the implementation rather than a live forward execution in this turn.

## 10. Training Pipeline

- Observed Fact: `PDBbindPairDataset` eagerly processes every selected complex in `__init__`, stores successful samples in memory, and records skipped samples (`protliggnn_train.py:500`, `protliggnn_train.py:503`, `protliggnn_train.py:507`).
- Observed Fact: `DataLoader` uses default worker settings with a custom collate function that builds PyG `Batch` objects (`protliggnn_train.py:528`, `protliggnn_train.py:705`).
- Observed Fact: Training uses `model.train(is_train)`, computes predictions, applies `F.mse_loss`, zeroes gradients, backpropagates, and steps Adam (`protliggnn_train.py:712`, `protliggnn_train.py:728`).
- Observed Fact: Early stopping uses validation RMSE if available, otherwise validation loss, with patience default 10 (`protliggnn_train.py:875`, `protliggnn_train.py:944`).
- Observed Fact: Checkpoints save only model state dict, args, and stats (`protliggnn_train.py:771`).
- Observed Fact: No learning-rate scheduler, gradient clipping, mixed precision, multi-GPU synchronization, distributed sampler, gradient accumulation, or optimizer-state checkpointing exists in the affinity trainer.

## 11. Loss Function Analysis

- Observed Fact: Primary affinity loss is MSE: `F.mse_loss(preds, labels)` (`protliggnn_train.py:728`).
- Mathematical form: `L = (1/B) * sum_i (yhat_i - y_i)^2`.
- Strong Inference: MSE is chosen because labels are continuous pAffinity values and RMSE/PCC are central evaluation metrics.
- Limitation: MSE overweights outliers and censored labels (`<`) are treated as exact values after parsing. There is no heteroscedastic, ranking, censor-aware, or uncertainty-aware objective.
- Observed Fact: The SMILES generator uses cross-entropy over next-character prediction with padding ignored (`train_smiles_generator.py:281`) and clips gradients at max norm 5 (`train_smiles_generator.py:150`).

## 12. Optimizer and Hyperparameters

Affinity trainer defaults:
- Observed Fact: epochs 50, batch size 8, lr 1e-3, weight decay 1e-5, patience 10, seed 42, device CPU, optional `--max_samples`, optional `--no_crossgraph` (`protliggnn_train.py:869`).
- Observed Fact: Optimizer is Adam with configured lr and weight decay (`protliggnn_train.py:916`).
- Strong Inference: Small batch size likely reflects graph memory constraints and CPU-friendly defaults.

SMILES generator defaults:
- Observed Fact: epochs 30, batch size 64, hidden size 64, 1 LSTM layer, lr 1e-3, seq_len 96 (`train_smiles_generator.py:71`).
- Observed Fact: README example recommends hidden size 256 and 2 layers, which differs from script defaults (`README.md:113`, `train_smiles_generator.py:73`).

## 13. Evaluation Methodology

- Observed Fact: Metrics are PCC, Spearman, RMSE, and MAE (`protliggnn_train.py:754`).
- Observed Fact: Test predictions are saved with `pdb_id`, true affinity, predicted affinity, and signed error (`protliggnn_train.py:814`).
- Observed Fact: Scatter and training-curve plots are generated after testing (`protliggnn_train.py:823`, `protliggnn_train.py:838`).
- Observed Fact: Report scripts aggregate `epoch_history_*.csv`, `test_predictions_*.csv`, logs, and comparison text files (`generate_paper_figures.py:24`, `generate_paper_figures.py:37`, `report_assets.py:25`).
- Limitation: No confidence intervals, bootstrap, multiple seeds, target/scaffold/time splits, external baseline reproduction, or significance testing.

## 14. Experiment Tracking

- Observed Fact: Run names are plain strings passed by `--run_name` and used in output filenames (`protliggnn_train.py:878`, `protliggnn_train.py:976`).
- Observed Fact: The final package expects files like `outputs/comparison_all_models_5000_e40.txt`, `outputs/genai/generated_molecule_quality_summary.json`, and Vina outputs (`final_research_package.py:11`, `final_research_package.py:15`, `final_research_package.py:17`).
- Observed Fact: Current worktree has compact final summaries but many referenced rich outputs are absent or deleted.
- Reproducibility Risk: Final summary metrics are retained, but exact model checkpoints, logs, split IDs, prediction CSVs, and generated molecule CSVs are not currently available in tracked/retained outputs.

## 15. Scientific Contributions

Actually novel in this implementation:
- Strong Inference: The integration workflow is the primary novelty: PDBBind graph predictor -> generated molecules -> ProtLigGNN score -> GAPS -> Vina.
- Observed Fact: GAPS is hand-defined as 0.50 normalized affinity + 0.20 novelty + 0.15 normalized QED + 0.15 diversity (`score_generated_with_protliggnn.py:146`, `score_generated_with_protliggnn.py:158`).
- Observed Fact: The `--no_crossgraph` pathway supports an ablation of cross-graph interaction (`protliggnn_train.py:641`, `protliggnn_train.py:879`).

Not novel by implementation:
- GAT, GCN, MultiheadAttention, RDKit descriptors, character-level LSTM generation, ETKDG/UFF 3D embedding, Open Babel conversion, and AutoDock Vina docking are standard components.

Surprising result:
- Observed Fact from retained summary: `no_crossgraph_5000_e40` outperformed `crossgraph_5000_e40` and `crossgraph_attention_5000_e40` on PCC/RMSE in the retained comparison (`outputs/final_research_package/final_results_summary.md:23`).
- Strong Inference: The cross-attention mechanism may not yet be beneficial, possibly due to small data, coarse features, mean pooling dilution, lack of edge geometry, or overfitting.

## 16. Mathematical Reconstruction

Label conversion:
```text
c = value * unit_scale
y = -log10(c)
```
implemented in `parse_affinity_to_pkd` (`protliggnn_train.py:160`).

Ligand graph:
```text
G_l = (V_l, E_l)
v_a = atom_features(atom_a) in R^78
E_l = {(i,j),(j,i) for each RDKit bond i-j}
```

Protein pocket graph:
```text
V_p = {residue r: min_atom_atom_distance(r, ligand) <= 6A}
v_r = residue_features(r) in R^30
E_p = {(i,j),(j,i): ||center_i - center_j||_2 <= 8A}
```

Encoders:
```text
H_l^1 = LN(ELU(GAT_1(X_l, E_l)))
H_l^2 = LN(ELU(GAT_2(H_l^1, E_l)))
H_l^3 = LN(ELU(GAT_3(H_l^2, E_l)))

H_p^1 = LN(ReLU(GCN_1(X_p, E_p)))
H_p^2 = LN(ReLU(GCN_2(H_p^1, E_p)))
H_p^3 = LN(ReLU(GCN_3(H_p^2, E_p)))
```

Cross-attention per complex:
```text
C_l = LN(MHA(Q=H_l, K=H_p, V=H_p) + H_l)
C_p = LN(MHA(Q=H_p, K=H_l, V=H_l) + H_p)
Z_l = concat(H_l, C_l)
Z_p = concat(H_p, C_p)
```

No-crossgraph ablation:
```text
Z_l = concat(H_l, 0)
Z_p = concat(H_p, 0)
```

Graph readout and regression:
```text
g_l = mean_nodes(Z_l)
g_p = mean_nodes(Z_p)
h = concat(g_l, g_p)
yhat = MLP(h)
```

GAPS:
```text
GAPS = 0.50 * minmax(pred_affinity)
     + 0.20 * novelty
     + 0.15 * minmax(QED)
     + 0.15 * diversity
diversity_i = 1 - max_{j != i} Tanimoto(fp_i, fp_j)
```

## 17. Code Architecture

- Observed Fact: There is no package/module hierarchy; files are scripts with import reuse.
- Observed Fact: `infer_affinity.py` imports model and graph-building helpers from `protliggnn_train.py` (`infer_affinity.py:8`).
- Observed Fact: `score_generated_with_protliggnn.py` imports inference helpers dynamically and optionally appends `python_deps` (`score_generated_with_protliggnn.py:60`).
- Observed Fact: `dock_top_candidates.py` imports Vina helper functions from `vina_validation.py` (`dock_top_candidates.py:7`).
- Strong Inference: The architecture evolved from notebooks/scripts into a runnable pipeline, not from a library-first design.

Execution dependency graph:

```text
protliggnn_train.py
  -> checkpoint, epoch_history, test_predictions, plots

infer_affinity.py
  -> imports protliggnn_train graph/model helpers

genai_extract_ligands.py
  -> pdbbind_ligands_smiles.csv
  -> train_smiles_generator.py
      -> smiles_lstm_generator.pt, vocab, history
      -> generate_and_filter_molecules.py
          -> generated_molecules.csv, quality summary
          -> score_generated_with_protliggnn.py
              -> top10 affinity/GAPS CSVs
              -> vina_validation.py / dock_top_candidates.py
                  -> docking CSVs, poses, final comparison

report_assets.py / top_affinity_predictions.py / generate_paper_figures.py
  -> report tables and figures from existing artifacts

final_research_package.py
  -> compact final Markdown/TXT/CSV summaries
```

## 18. File-by-File Breakdown

- `protliggnn_train.py`: Core research implementation. Parses PDBBind labels, builds ligand/protein graphs, defines `ProtLigGNN`, trains/evaluates, saves checkpoint/history/predictions/plots.
- `infer_affinity.py`: Single protein-ligand inference path. Loads checkpoint, rebuilds graph pair, predicts scalar affinity, writes JSON (`infer_affinity.py:27`, `infer_affinity.py:40`, `infer_affinity.py:81`).
- `genai_extract_ligands.py`: Extracts unique canonical SMILES from all local SDF/MOL2 ligand files (`genai_extract_ligands.py:13`, `genai_extract_ligands.py:38`).
- `train_smiles_generator.py`: Character-level next-token LSTM trainer for ligand SMILES (`train_smiles_generator.py:28`, `train_smiles_generator.py:51`, `train_smiles_generator.py:281`).
- `generate_and_filter_molecules.py`: Samples from the LSTM, canonicalizes with RDKit, computes QED/MW/LogP/TPSA/HBD/HBA/Lipinski, writes generated molecule CSV and quality summary (`generate_and_filter_molecules.py:99`, `generate_and_filter_molecules.py:135`, `generate_and_filter_molecules.py:163`).
- `score_generated_with_protliggnn.py`: Converts generated SMILES to 3D SDF, scores with ProtLigGNN, computes diversity and GAPS, writes ranked top candidates (`score_generated_with_protliggnn.py:76`, `score_generated_with_protliggnn.py:98`, `score_generated_with_protliggnn.py:146`).
- `vina_validation.py`: Runs Open Babel and AutoDock Vina, infers receptor boxes, parses Vina scores, validates either one ligand or generated batch (`vina_validation.py:41`, `vina_validation.py:129`, `vina_validation.py:147`, `vina_validation.py:196`).
- `dock_top_candidates.py`: Batch docking wrapper for known top-10 predictions and generated top-10 GAPS candidates, hard-wired to 2l3r as generated target (`dock_top_candidates.py:19`, `dock_top_candidates.py:21`, `dock_top_candidates.py:91`, `dock_top_candidates.py:138`).
- `top_affinity_predictions.py`: Aggregates `test_predictions_*.csv` and exports the top predicted known binders (`top_affinity_predictions.py:15`, `top_affinity_predictions.py:27`).
- `report_assets.py`: Parses logs/results and creates report tables, ablation plot, training loss plot, prediction scatter, and summary (`report_assets.py:25`, `report_assets.py:74`, `report_assets.py:96`).
- `generate_paper_figures.py`: Builds paper-style figures from existing CSV/text artifacts (`generate_paper_figures.py:24`, `generate_paper_figures.py:37`, `generate_paper_figures.py:220`).
- `final_research_package.py`: Builds compact final summary files from existing outputs (`final_research_package.py:11`, `final_research_package.py:337`).
- `requirements.txt`: Minimal unpinned dependency names. No versions.
- `.gitignore`: Excludes data, checkpoints, figures, logs, local envs, and most outputs, while allowing compact final summaries.

## 19. Performance and Scalability Audit

Strengths:
- Observed Fact: Graphs are sparse PyG `Data` objects, not dense voxel grids.
- Observed Fact: Pocket extraction limits protein graph size to residues near ligand atoms.

Bottlenecks:
- Observed Fact: Dataset construction is eager and single-process, with RDKit and BioPython parsing all selected complexes before training starts (`protliggnn_train.py:500`).
- Observed Fact: DataLoader does not set `num_workers`, `pin_memory`, or persistent workers.
- Observed Fact: Cross-attention loops over every complex in a batch in Python (`protliggnn_train.py:656`), causing poor GPU utilization for many small graphs.
- Observed Fact: Cross-attention complexity is O(N_l*N_p) per complex and attention maps are not reused or logged.
- Observed Fact: No mixed precision or distributed training support.

Recommendations:
- Cache processed graph pairs to disk with versioned preprocessing config.
- Add edge attributes: bond type, distance, residue contact distance, ligand-protein geometric distances.
- Use PyG batching-friendly cross-graph interaction or sparse bipartite edges instead of per-sample Python attention loops.
- Add `num_workers`, `pin_memory`, and prefetching where platform permits.
- Add AMP for CUDA runs and save optimizer/scheduler state for resumability.

## 20. Reproducibility Audit

Positive:
- Observed Fact: Seeds are set for Python, NumPy, torch, and CUDA if available (`protliggnn_train.py:138`).
- Observed Fact: Checkpoints include model args and stats (`protliggnn_train.py:771`).
- Observed Fact: Output filenames include run names.

Missing or weak:
- Observed Fact: Dependencies are unpinned in `requirements.txt`.
- Observed Fact: The local `venv` is broken/non-portable on this machine.
- Observed Fact: No split manifest is saved, so exact train/val/test PDB IDs are not retained.
- Observed Fact: Checkpoints do not include optimizer state, RNG state, or preprocessing version.
- Observed Fact: No automated tests were found.
- Observed Fact: Current retained outputs lack the underlying rich artifacts used to generate final metrics.
- Reproducibility Risk: The PDBBind index README says structure files are PDBbind v2024 reprocessed files associated with v2020 entries, latest update Aug 4, 2025 (`data/pdbbind2020/index/README:4`, `data/pdbbind2020/index/README:9`). This should be stated precisely in any paper.

## 21. Security and Robustness Review

- Observed Fact: External commands use `subprocess.run(command_list, ...)` rather than string shell commands, which reduces injection risk (`vina_validation.py:35`, `vina_validation.py:98`, `vina_validation.py:147`).
- Observed Fact: Path inputs are user-provided in inference and docking scripts with existence checks.
- Robustness Gap: There is no sandboxing for external binaries, no output validation beyond file existence/size for Open Babel conversion, and docking boxes are inferred blindly from full receptor extents.
- Robustness Gap: Invalid molecules and docking failures are often skipped, which is pragmatic, but summary statistics depend on skipped counts being preserved.

## 22. Technical Debt

- Single large training script mixes data parsing, featurization, model definitions, training, metrics, and plotting.
- No reusable package layout or typed config.
- No test suite or smoke-test fixture.
- No cached preprocessing.
- No model registry or artifact manifest.
- No baseline implementations in code beyond `--no_crossgraph`.
- No clear separation between exploratory generated artifacts and reproducible experiment outputs.
- Retained final metrics are not backed by current checkpoints/prediction CSVs in the visible output tree.

## 23. Improvement Opportunities

High priority:
- Fix affinity parser to support `IC50`, `EC50`, and censored values with appropriate censor-aware treatment.
- Save train/val/test PDB ID manifests and evaluate with scaffold, target-family, and temporal splits.
- Add graph preprocessing cache and a small test fixture.
- Pin dependencies and provide a working environment file.
- Add explicit baselines: ligand-only, protein-only, GCN-only, GAT-only, RF/XGBoost descriptors, docking score baseline, and known literature baselines.

Modeling:
- Add bond and distance edge attributes.
- Incorporate 3D geometry via equivariant GNNs, distance-aware attention, or protein-ligand bipartite contact graphs.
- Replace global mean pooling with attention/readout pooling or interaction-site pooling.
- Expose and analyze attention maps if cross-attention remains a contribution.
- Add uncertainty estimation or conformal prediction for screening decisions.

GenAI:
- Current generator is target-agnostic. Add target-conditioned generation or active learning.
- Use stronger generative models: graph generator, SMILES transformer, diffusion, or fragment-based methods.
- Penalize synthetic accessibility/reactive groups and add medicinal chemistry filters beyond Lipinski/QED.

## 24. Publication Readiness Review

Reviewer-style scores, based on implementation evidence:

| Category | Score / 10 | Rationale |
|---|---:|---|
| Novelty | 4 | Integration is useful, but core models and tools are standard; GAPS is a hand-weighted score. |
| Technical Quality | 5 | Pipeline is coherent, but geometry/edge attributes, split rigor, and parser limitations are major issues. |
| Experimental Design | 3 | Random split only; no strong baselines, no statistical tests, missing artifacts. |
| Engineering Quality | 5 | Scripts are runnable in principle, but not packaged, not tested, and not scalable. |
| Reproducibility | 3 | Seeds exist, but env is unpinned/broken, artifacts are missing, split manifests absent. |
| Publication Readiness | 3 | Needs benchmark rigor, artifact cleanup, baselines, ablations, and clearer claims. |

Likely top-tier reviewer verdict:
- Strength: Clear end-to-end idea and implementation spanning affinity prediction, generation, filtering, and docking.
- Weakness: Experimental rigor is not yet sufficient to support strong scientific claims.
- Main concern: The retained best result says no-crossgraph outperforms crossgraph/cross-attention, which undermines any claim that the proposed cross-graph interaction is the key modeling contribution.

## 25. Compare Against Documentation

Agreement:
- Observed Fact: README describes the closed-loop workflow, GAT ligand encoder, GCN protein encoder, SMILES LSTM, RDKit filtering, GAPS, and Vina validation; all are implemented (`README.md:3`, `README.md:9`, `README.md:13`, `README.md:25`).
- Observed Fact: Final summary correctly states the integration novelty rather than claiming RDKit/Vina novelty (`outputs/final_research_package/final_results_summary.md:5`, `outputs/final_research_package/final_results_summary.md:7`).
- Observed Fact: README reference metrics match final summary metrics (`README.md:169`, `outputs/final_research_package/final_results_summary.md:13`).

Contradictions or caveats:
- Observed Fact: README says "The dataset is not included in this repository" (`README.md:68`), but the local workspace contains `data/pdbbind2020` with 12,235 complex directories. This may be true for Git tracking due `.gitignore`, but false for the current working copy.
- Observed Fact: README frames data as PDBbind v2020 (`README.md:55`), while the local data README says v2020 entries with PDBbind v2024 reprocessed structure files and updated metadata (`data/pdbbind2020/index/README:4`, `data/pdbbind2020/index/README:9`).
- Observed Fact: README's example best run is `--no_crossgraph`, and retained results show no-crossgraph best. Therefore any paper claim that bidirectional cross-attention improves performance is not supported by the retained implementation evidence.
- Observed Fact: The final package references source evidence files such as `outputs/comparison_all_models_5000_e40.txt` and generated/docking CSVs, but those richer files are not currently present in `outputs/` outside compact summaries.

## 26. Future Research Directions

- Build a rigorous PDBBind benchmark suite with fixed splits, multiple seeds, and external baselines.
- Upgrade the affinity model to geometry-aware equivariant or distance-aware message passing.
- Treat censored affinity values properly.
- Add target-conditioned molecular generation.
- Convert GAPS from hand weights to validated multi-objective optimization or Bayesian acquisition.
- Evaluate generated candidates with retrosynthesis, PAINS/reactivity filters, docking with pocket-specific boxes, rescoring, MD, and eventually experimental assays.

## 27. Bottom Line

Strong Inference: This is a solid prototype of an integrated protein-ligand GNN plus generative screening pipeline. The implementation supports the claim that the project predicts PDBBind affinity from paired graphs and uses that model to prioritize generated ligands. It does not yet support strong claims of state-of-the-art modeling, robust cross-graph attention benefit, or publication-grade experimental rigor. The next version should focus less on adding features and more on reproducible benchmarking, geometry-aware modeling, parser correctness, artifact traceability, and scientifically defensible validation.
