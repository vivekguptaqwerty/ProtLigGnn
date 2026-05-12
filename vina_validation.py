import argparse
import json
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch_geometric.data import Batch

from infer_affinity import build_graph_pair, load_model


WINDOWS_VINA_INSTRUCTIONS = """AutoDock Vina was not found on PATH.

Windows installation options:
1. Download a Windows build of AutoDock Vina from the official release page.
2. Extract the archive and add the folder containing `vina.exe` to your PATH.
3. Open a new terminal and verify with: `vina --version`

You can also place `vina.exe` in a directory already on PATH, then rerun this utility.
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a ProtLigGNN prediction and prepare docking outputs.")
    parser.add_argument("--protein_pdb", type=str, required=True)
    parser.add_argument("--ligand_file", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="outputs/best_protliggnn.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out_dir", type=str, default="outputs/vina_validation")
    return parser.parse_args()


def detect_vina() -> Tuple[bool, Optional[str]]:
    try:
        result = subprocess.run(
            ["vina", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        version_text = (result.stdout or result.stderr).strip()
        return True, version_text
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False, None


def run_protliggnn_inference(protein_pdb: Path, ligand_file: Path, checkpoint_path: Path, device: torch.device) -> float:
    model = load_model(checkpoint_path, device)
    ligand_graph, protein_graph = build_graph_pair(protein_pdb, ligand_file)
    ligand_batch = Batch.from_data_list([ligand_graph]).to(device)
    protein_batch = Batch.from_data_list([protein_graph]).to(device)
    with torch.no_grad():
        return float(model(ligand_batch, protein_batch).item())


def save_outputs(out_dir: Path, result_payload: dict, summary_text: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "vina_result.json"
    summary_path = out_dir / "validation_summary.txt"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result_payload, handle, indent=2)
    summary_path.write_text(summary_text, encoding="utf-8")


def main():
    args = parse_args()
    protein_pdb = Path(args.protein_pdb)
    ligand_file = Path(args.ligand_file)
    checkpoint_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    device = torch.device(args.device)

    if not protein_pdb.exists():
        raise FileNotFoundError(f"Protein PDB not found: {protein_pdb}")
    if not ligand_file.exists():
        raise FileNotFoundError(f"Ligand file not found: {ligand_file}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    protliggnn_predicted_affinity = run_protliggnn_inference(
        protein_pdb=protein_pdb,
        ligand_file=ligand_file,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    vina_available, vina_version = detect_vina()
    docking_status = "vina_available" if vina_available else "vina_not_installed"
    vina_score = None

    # This JSON schema is intentionally ready for future docking integration.
    # Once Vina preprocessing is added, this block can be extended with receptor/ligand
    # preparation paths, docking box parameters, pose files, and parsed Vina scores.
    result_payload = {
        "protein_pdb": str(protein_pdb),
        "ligand_file": str(ligand_file),
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "protliggnn_predicted_affinity": protliggnn_predicted_affinity,
        "vina_score": vina_score,
        "docking_status": docking_status,
        "vina_version": vina_version,
    }

    summary_lines = [
        "ProtLigGNN + Vina Validation Summary",
        "===================================",
        "",
        f"Protein PDB: {protein_pdb}",
        f"Ligand file: {ligand_file}",
        f"Checkpoint: {checkpoint_path}",
        f"Device: {device}",
        f"ProtLigGNN predicted affinity: {protliggnn_predicted_affinity:.4f}",
        f"Docking status: {docking_status}",
        f"Vina score: {vina_score}",
    ]
    if vina_version:
        summary_lines.append(f"Vina version: {vina_version}")
    else:
        summary_lines.extend(
            [
                "",
                "AutoDock Vina is not installed or not available on PATH.",
                "Future docking integration path:",
                "- prepare receptor and ligand PDBQT files",
                "- define docking box around the binding pocket",
                "- run `vina` and parse the reported best score",
                "- update this summary with real docking outputs",
            ]
        )
    summary_text = "\n".join(summary_lines) + "\n"

    save_outputs(out_dir, result_payload, summary_text)

    print(f"ProtLigGNN predicted affinity: {protliggnn_predicted_affinity:.4f}")
    print(f"Docking status: {docking_status}")
    print(f"Saved validation result to {(out_dir / 'vina_result.json').resolve()}")
    print(f"Saved summary to {(out_dir / 'validation_summary.txt').resolve()}")

    if not vina_available:
        print("")
        print(WINDOWS_VINA_INSTRUCTIONS)


if __name__ == "__main__":
    main()
