import argparse
import csv
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem


INPUT_CSV = Path("outputs") / "genai" / "generated_molecules.csv"
OUTPUT_DIR = Path("outputs") / "genai"
LOCAL_DEPS_DIR = Path("python_deps")
SCORED_CSV = OUTPUT_DIR / "generated_molecules_scored.csv"
TOP10_AFFINITY_CSV = OUTPUT_DIR / "top10_generated_candidates.csv"
TOP10_GAPS_CSV = OUTPUT_DIR / "top10_generated_candidates_by_gaps.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Score generated molecules with trained ProtLigGNN.")
    parser.add_argument("--protein_pdb", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="outputs/best_protliggnn.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_molecules", type=int, default=None)
    return parser.parse_args()


def load_generated_molecules(path: Path, max_molecules: Optional[int]) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Generated molecules CSV not found: {path}")

    rows = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "smiles",
            "qed",
            "mol_weight",
            "logp",
            "tpsa",
            "hbd",
            "hba",
            "lipinski_violations",
            "is_novel",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

        for row in reader:
            rows.append(row)
            if max_molecules is not None and len(rows) >= max_molecules:
                break

    return rows


def load_protliggnn_helpers():
    if LOCAL_DEPS_DIR.exists():
        sys.path.append(str(LOCAL_DEPS_DIR.resolve()))

    try:
        from torch_geometric.data import Batch
        from infer_affinity import build_graph_pair, load_model
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Unable to import the ProtLigGNN inference stack. "
            "Install the missing dependency shown in the original error, then rerun scoring."
        ) from exc

    return Batch, build_graph_pair, load_model


def smiles_to_temp_sdf(smiles: str, output_path: Path) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    mol = Chem.AddHs(mol)
    embed_status = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if embed_status != 0:
        return False

    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass

    mol.SetProp("_Name", smiles)
    writer = Chem.SDWriter(str(output_path))
    writer.write(mol)
    writer.close()
    return output_path.exists() and output_path.stat().st_size > 0


def predict_affinity(model, batch_cls, build_graph_pair_fn, protein_pdb: Path, ligand_sdf: Path, device: torch.device) -> float:
    ligand_graph, protein_graph = build_graph_pair_fn(protein_pdb, ligand_sdf)
    ligand_batch = batch_cls.from_data_list([ligand_graph]).to(device)
    protein_batch = batch_cls.from_data_list([protein_graph]).to(device)

    with torch.no_grad():
        return float(model(ligand_batch, protein_batch).item())


def min_max_normalize(values: Sequence[float]) -> List[float]:
    if not values:
        return []

    min_value = min(values)
    max_value = max(values)
    if max_value == min_value:
        return [1.0 for _ in values]

    return [(value - min_value) / (max_value - min_value) for value in values]


def molecule_fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def compute_diversity_scores(rows: Sequence[Dict[str, object]]) -> List[float]:
    fingerprints = [molecule_fingerprint(str(row["smiles"])) for row in rows]
    scores = []

    for idx, fingerprint in enumerate(fingerprints):
        if fingerprint is None:
            scores.append(0.0)
            continue

        similarities = []
        for other_idx, other_fingerprint in enumerate(fingerprints):
            if idx == other_idx or other_fingerprint is None:
                continue
            similarities.append(DataStructs.TanimotoSimilarity(fingerprint, other_fingerprint))

        scores.append(1.0 - max(similarities) if similarities else 1.0)

    return scores


def add_gaps_scores(rows: List[Dict[str, object]]):
    affinity_scores = [float(row["predicted_affinity"]) for row in rows]
    qed_scores = [float(row["qed"]) for row in rows]
    normalized_affinity = min_max_normalize(affinity_scores)
    normalized_qed = min_max_normalize(qed_scores)
    diversity_scores = compute_diversity_scores(rows)

    for idx, row in enumerate(rows):
        novelty_score = 1.0 if str(row["is_novel"]).lower() == "true" else 0.0
        # GAPS = Generative Affinity Prioritization Score.
        # It combines ProtLigGNN affinity, novelty, QED, and generated-set diversity.
        row["gaps"] = (
            0.50 * normalized_affinity[idx]
            + 0.20 * novelty_score
            + 0.15 * normalized_qed[idx]
            + 0.15 * diversity_scores[idx]
        )


def save_rows(rows: Sequence[Dict[str, object]], path: Path, include_gaps: bool = False):
    fieldnames = [
        "smiles",
        "predicted_affinity",
        "qed",
        "mol_weight",
        "logp",
        "tpsa",
        "hbd",
        "hba",
        "lipinski_violations",
        "is_novel",
    ]
    if include_gaps:
        fieldnames.append("gaps")

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    RDLogger.DisableLog("rdApp.*")

    protein_pdb = Path(args.protein_pdb)
    checkpoint = Path(args.checkpoint)
    device = torch.device(args.device)

    if not protein_pdb.exists():
        raise FileNotFoundError(f"Protein PDB not found: {protein_pdb}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"ProtLigGNN checkpoint not found: {checkpoint}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generated_rows = load_generated_molecules(INPUT_CSV, args.max_molecules)
    batch_cls, build_graph_pair_fn, load_model_fn = load_protliggnn_helpers()
    model = load_model_fn(checkpoint, device)

    scored_rows = []
    skipped = 0

    with tempfile.TemporaryDirectory(prefix="protliggnn_generated_") as temp_dir:
        temp_root = Path(temp_dir)
        for idx, row in enumerate(generated_rows, start=1):
            smiles = row["smiles"]
            ligand_sdf = temp_root / f"generated_{idx:06d}.sdf"

            if not smiles_to_temp_sdf(smiles, ligand_sdf):
                skipped += 1
                continue

            try:
                predicted_affinity = predict_affinity(
                    model,
                    batch_cls,
                    build_graph_pair_fn,
                    protein_pdb,
                    ligand_sdf,
                    device,
                )
            except Exception as exc:
                skipped += 1
                print(f"skipping {smiles}: {exc}")
                continue

            scored_rows.append(
                {
                    "smiles": smiles,
                    "predicted_affinity": predicted_affinity,
                    "qed": row["qed"],
                    "mol_weight": row["mol_weight"],
                    "logp": row["logp"],
                    "tpsa": row["tpsa"],
                    "hbd": row["hbd"],
                    "hba": row["hba"],
                    "lipinski_violations": row["lipinski_violations"],
                    "is_novel": row["is_novel"],
                }
            )

            if idx % 25 == 0 or idx == len(generated_rows):
                print(f"processed {idx}/{len(generated_rows)} molecules")

    if not scored_rows:
        raise RuntimeError("No generated molecules could be scored.")

    add_gaps_scores(scored_rows)

    by_affinity = sorted(scored_rows, key=lambda row: float(row["predicted_affinity"]), reverse=True)
    by_gaps = sorted(scored_rows, key=lambda row: float(row["gaps"]), reverse=True)

    save_rows(by_affinity, SCORED_CSV, include_gaps=False)
    save_rows(by_affinity[:10], TOP10_AFFINITY_CSV, include_gaps=False)
    save_rows(by_gaps[:10], TOP10_GAPS_CSV, include_gaps=True)

    print(f"loaded generated molecules: {len(generated_rows)}")
    print(f"scored molecules: {len(scored_rows)}")
    print(f"skipped molecules: {skipped}")
    print("top candidates by predicted affinity:")
    for row in by_affinity[:10]:
        print(f"{row['smiles']} | affinity={float(row['predicted_affinity']):.4f} | QED={float(row['qed']):.3f}")
    print(f"saved scored molecules: {SCORED_CSV}")
    print(f"saved top 10 by affinity: {TOP10_AFFINITY_CSV}")
    print(f"saved top 10 by GAPS: {TOP10_GAPS_CSV}")


if __name__ == "__main__":
    main()
