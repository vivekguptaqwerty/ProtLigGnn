import argparse
import json
from pathlib import Path

import torch
from torch_geometric.data import Batch

from protliggnn_train import (
    LIGAND_FEATURE_DIM,
    PROTEIN_FEATURE_DIM,
    ProtLigGNN,
    build_protein_pocket_graph,
    load_ligand_molecule,
    mol_to_ligand_graph,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run ProtLigGNN inference for one protein-ligand pair.")
    parser.add_argument("--protein_pdb", type=str, required=True)
    parser.add_argument("--ligand_file", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="outputs/best_protliggnn.pt")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> ProtLigGNN:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_args = checkpoint.get("args", {})
    model = ProtLigGNN(
        ligand_dim=LIGAND_FEATURE_DIM,
        protein_dim=PROTEIN_FEATURE_DIM,
        no_crossgraph=bool(checkpoint_args.get("no_crossgraph", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def build_graph_pair(protein_pdb: Path, ligand_file: Path):
    ligand_mol = load_ligand_molecule(ligand_file)
    if ligand_mol is None:
        raise ValueError(f"Unable to read ligand file or missing 3D conformer: {ligand_file}")

    ligand_graph = mol_to_ligand_graph(ligand_mol)
    protein_graph = build_protein_pocket_graph(protein_pdb, ligand_graph.pos.numpy())
    return ligand_graph, protein_graph


def main():
    args = parse_args()
    protein_pdb = Path(args.protein_pdb)
    ligand_file = Path(args.ligand_file)
    checkpoint_path = Path(args.checkpoint)
    device = torch.device(args.device)

    if not protein_pdb.exists():
        raise FileNotFoundError(f"Protein PDB not found: {protein_pdb}")
    if not ligand_file.exists():
        raise FileNotFoundError(f"Ligand file not found: {ligand_file}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = load_model(checkpoint_path, device)
    ligand_graph, protein_graph = build_graph_pair(protein_pdb, ligand_file)

    ligand_batch = Batch.from_data_list([ligand_graph]).to(device)
    protein_batch = Batch.from_data_list([protein_graph]).to(device)

    with torch.no_grad():
        predicted_affinity = float(model(ligand_batch, protein_batch).item())

    output = {
        "protein_pdb": str(protein_pdb),
        "ligand_file": str(ligand_file),
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "predicted_affinity": predicted_affinity,
    }

    output_path = Path("outputs") / "inference_result.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print(f"Predicted binding affinity: {predicted_affinity:.4f}")
    print(f"Saved inference result to {output_path.resolve()}")


if __name__ == "__main__":
    main()
