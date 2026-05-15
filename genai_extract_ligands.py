import csv
from pathlib import Path
from typing import Iterable, Optional

from rdkit import Chem, RDLogger


DATA_ROOT = Path("data") / "pdbbind2020"
OUTPUT_PATH = Path("outputs") / "genai" / "pdbbind_ligands_smiles.csv"
LIGAND_EXTENSIONS = {".sdf", ".mol2"}


def iter_ligand_files(data_root: Path) -> Iterable[Path]:
    for path in data_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in LIGAND_EXTENSIONS:
            yield path


def load_ligand(path: Path) -> Optional[Chem.Mol]:
    suffix = path.suffix.lower()

    try:
        if suffix == ".sdf":
            supplier = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=False)
            for mol in supplier:
                if mol is not None:
                    return mol
            return None

        if suffix == ".mol2":
            return Chem.MolFromMol2File(str(path), sanitize=True, removeHs=False)
    except Exception:
        return None

    return None


def mol_to_canonical_smiles(mol: Chem.Mol) -> Optional[str]:
    try:
        smiles = Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None

    return smiles if smiles else None


def main():
    RDLogger.DisableLog("rdApp.*")

    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"PDBbind data root not found: {DATA_ROOT}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    ligand_files = sorted(iter_ligand_files(DATA_ROOT))
    total_found = len(ligand_files)
    valid_extracted = 0
    invalid_skipped = 0
    duplicates_removed = 0

    seen_smiles = set()
    rows = []

    for ligand_file in ligand_files:
        mol = load_ligand(ligand_file)
        smiles = mol_to_canonical_smiles(mol) if mol is not None else None

        if smiles is None:
            invalid_skipped += 1
            continue

        valid_extracted += 1
        if smiles in seen_smiles:
            duplicates_removed += 1
            continue

        seen_smiles.add(smiles)
        rows.append(
            {
                "pdb_id": ligand_file.parent.name,
                "ligand_file": str(ligand_file),
                "smiles": smiles,
            }
        )

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pdb_id", "ligand_file", "smiles"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"total ligand files found: {total_found}")
    print(f"valid molecules extracted: {valid_extracted}")
    print(f"invalid skipped: {invalid_skipped}")
    print(f"duplicate SMILES removed: {duplicates_removed}")
    print(f"final unique SMILES count: {len(rows)}")
    print(f"saved CSV: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
