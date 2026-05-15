# ProtLigGNN

ProtLigGNN is a graph neural network pipeline for protein-ligand binding affinity prediction, extended with a lightweight generative chemistry loop for ligand proposal, filtering, affinity-guided prioritization, and docking-based secondary validation.

The core contribution is a closed-loop workflow:

```text
PDBbind complexes -> ProtLigGNN affinity model
PDBbind ligands -> SMILES generator -> RDKit filtering
generated ligands -> ProtLigGNN scoring -> GAPS ranking -> AutoDock Vina validation
```

RDKit and AutoDock Vina are used as chemistry and docking validation tools. They are not the novelty of the project. The novelty is the integration of generative ligand design with a trained protein-ligand GNN for affinity-guided candidate prioritization.

## Highlights

- Protein-ligand affinity prediction from paired molecular graphs.
- Ligand graph construction with RDKit.
- Protein pocket graph construction with BioPython.
- Ligand encoder based on GATConv layers.
- Protein encoder based on GCNConv layers.
- Lightweight character-level SMILES LSTM generator.
- RDKit validation and descriptor filtering for generated molecules.
- ProtLigGNN-based scoring of generated candidates.
- GAPS ranking: Generative Affinity Prioritization Score.
- AutoDock Vina validation for known and generated top candidates.

## Repository Structure

```text
protliggnn_train.py              Train ProtLigGNN on PDBbind complexes
infer_affinity.py                Run affinity inference for one protein-ligand pair

genai_extract_ligands.py         Extract canonical SMILES from PDBbind ligands
train_smiles_generator.py        Train the character-level SMILES LSTM
generate_and_filter_molecules.py Generate and RDKit-filter candidate molecules
score_generated_with_protliggnn.py
                                 Score generated molecules with ProtLigGNN

vina_validation.py               Single and batch AutoDock Vina validation
dock_top_candidates.py           Dock known and generated top-10 candidate sets
final_research_package.py        Build compact final research summaries

report_assets.py                 Build tabular/visual reporting assets
top_affinity_predictions.py      Export top predicted known binders
generate_paper_figures.py        Generate paper-oriented figures

requirements.txt                 Python dependencies
```

Large datasets, trained checkpoints, docking poses, generated logs, and local dependency folders are intentionally excluded from version control.

## Data Layout

The training and extraction scripts expect PDBbind v2020 under:

```text
data/pdbbind2020/
```

Expected key paths:

```text
data/pdbbind2020/index/INDEX_general_PL.2020R1.lst
data/pdbbind2020/pdbbind_subset/
```

The dataset is not included in this repository.

## Installation

Create an isolated Python environment and install the Python dependencies:

```bash
pip install -r requirements.txt
```

For docking validation, install these external tools and ensure they are available from the shell:

```text
Open Babel
AutoDock Vina
```

On Windows, `vina_validation.py` also checks the common Vina installation path used by the Scripps Research Institute installer.

## ProtLigGNN Workflow

Train the affinity model:

```bash
python protliggnn_train.py --data_dir data/pdbbind2020 --device cpu --epochs 40 --max_samples 5000 --no_crossgraph --run_name no_crossgraph_5000_e40
```

Run inference for one complex:

```bash
python infer_affinity.py \
  --protein_pdb path/to/protein.pdb \
  --ligand_file path/to/ligand.sdf \
  --checkpoint outputs/best_protliggnn.pt \
  --device cpu
```

## GenAI Workflow

Extract ligand SMILES from PDBbind:

```bash
python genai_extract_ligands.py
```

Train the SMILES generator:

```bash
python train_smiles_generator.py --epochs 30 --batch_size 64 --hidden_size 256 --num_layers 2 --lr 0.001
```

Generate and filter candidate molecules:

```bash
python generate_and_filter_molecules.py --num_samples 10000 --temperature 0.8
```

Score generated molecules with ProtLigGNN:

```bash
python score_generated_with_protliggnn.py \
  --protein_pdb data/pdbbind2020/pdbbind_subset/2011-2019/2l3r/2l3r_protein.pdb \
  --checkpoint outputs/best_protliggnn.pt \
  --device cpu \
  --max_molecules 100
```

## Docking Validation

Validate one ligand SMILES with Vina:

```bash
python vina_validation.py \
  --protein_pdb data/pdbbind2020/pdbbind_subset/2011-2019/2l3r/2l3r_protein.pdb \
  --ligand_smiles "COc1ccccc1"
```

Dock the top known and generated candidate sets:

```bash
python dock_top_candidates.py
```

## Final Reporting

Build compact final research assets:

```bash
python final_research_package.py
```

This writes summary files under:

```text
outputs/final_research_package/
```

Only compact final summaries from this folder are allowed through `.gitignore`; large generated artifacts remain local.

## Current Reference Results

Best recorded ProtLigGNN run:

```text
no_crossgraph_5000_e40
PCC: 0.6714
Spearman: 0.6624
RMSE: 1.3722
MAE: 1.0677
```

Generated molecule quality from the completed 10,000-sample run:

```text
valid molecules: 6056
unique valid molecules: 4151
novel unique molecules: 3894
validity: 60.56%
novelty: 93.81%
```

These values are generated artifacts, not hard-coded assumptions. Re-run the workflows above to reproduce or update them.

## Notes on Version Control

The repository is configured to track source code and compact final summaries while excluding:

- PDBbind data
- model checkpoints
- local virtual environments
- local dependency folders
- generated docking poses
- generated logs and figures
- manuscript drafts and exported documents

This keeps the GitHub repository focused on the reproducible pipeline rather than local experiment byproducts.
