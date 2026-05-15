import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors


OUTPUT_DIR = Path("outputs") / "genai"
MODEL_PATH = OUTPUT_DIR / "smiles_lstm_generator.pt"
VOCAB_PATH = OUTPUT_DIR / "smiles_vocab.json"
TRAINING_SMILES_PATH = OUTPUT_DIR / "pdbbind_ligands_smiles.csv"
GENERATED_CSV_PATH = OUTPUT_DIR / "generated_molecules.csv"
SUMMARY_PATH = OUTPUT_DIR / "generated_molecule_quality_summary.json"


class SmilesLSTMGenerator(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int, embedding_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        embedded = self.embedding(input_ids)
        outputs, hidden = self.lstm(embedded, hidden)
        return self.output(outputs), hidden


def parse_args():
    parser = argparse.ArgumentParser(description="Generate and RDKit-filter molecules from the SMILES LSTM.")
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--max_length", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    return parser.parse_args()


def load_vocab(path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    char_to_idx = {str(char): int(idx) for char, idx in payload["char_to_idx"].items()}
    idx_to_char = {int(idx): str(char) for idx, char in payload["idx_to_char"].items()}
    return char_to_idx, idx_to_char


def load_model(path: Path, vocab_size: int) -> SmilesLSTMGenerator:
    if not path.exists():
        raise FileNotFoundError(f"Generator checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint["model_config"]
    model = SmilesLSTMGenerator(
        vocab_size=vocab_size,
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_layers"]),
        embedding_dim=int(config.get("embedding_dim", 64)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_training_smiles(path: Path) -> set:
    if not path.exists():
        raise FileNotFoundError(f"Training SMILES CSV not found: {path}")

    smiles_set = set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "smiles" not in reader.fieldnames:
            raise ValueError(f"CSV must contain a 'smiles' column: {path}")

        for row in reader:
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol is not None:
                smiles_set.add(Chem.MolToSmiles(mol, canonical=True))

    return smiles_set


def sample_smiles(
    model: SmilesLSTMGenerator,
    char_to_idx: Dict[str, int],
    idx_to_char: Dict[int, str],
    max_length: int,
    temperature: float,
) -> str:
    if temperature <= 0:
        raise ValueError("--temperature must be greater than 0")

    bos_idx = char_to_idx["<BOS>"]
    eos_idx = char_to_idx["<EOS>"]
    pad_idx = char_to_idx["<PAD>"]

    current = torch.tensor([[bos_idx]], dtype=torch.long)
    hidden = None
    chars = []

    with torch.no_grad():
        for _ in range(max_length):
            logits, hidden = model(current, hidden)
            next_logits = logits[:, -1, :] / temperature
            next_logits[:, pad_idx] = -float("inf")
            next_logits[:, bos_idx] = -float("inf")
            probabilities = F.softmax(next_logits, dim=-1)
            next_idx = int(torch.multinomial(probabilities, num_samples=1).item())

            if next_idx == eos_idx:
                break

            chars.append(idx_to_char[next_idx])
            current = torch.tensor([[next_idx]], dtype=torch.long)

    return "".join(chars)


def canonicalize_smiles(smiles: str) -> Optional[Tuple[str, Chem.Mol]]:
    if not smiles:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        canonical = Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None

    if not canonical:
        return None

    return canonical, mol


def lipinski_violations(mol: Chem.Mol) -> int:
    mol_weight = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)

    return int(mol_weight > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)


def molecule_row(smiles: str, mol: Chem.Mol, training_smiles: set) -> Dict[str, object]:
    return {
        "smiles": smiles,
        "qed": QED.qed(mol),
        "mol_weight": Descriptors.MolWt(mol),
        "logp": Crippen.MolLogP(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "hbd": Lipinski.NumHDonors(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "lipinski_violations": lipinski_violations(mol),
        "is_novel": smiles not in training_smiles,
    }


def save_generated(rows: Sequence[Dict[str, object]], path: Path):
    fieldnames = [
        "smiles",
        "qed",
        "mol_weight",
        "logp",
        "tpsa",
        "hbd",
        "hba",
        "lipinski_violations",
        "is_novel",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary(summary: Dict[str, object], path: Path):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main():
    args = parse_args()
    RDLogger.DisableLog("rdApp.*")
    torch.manual_seed(42)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    char_to_idx, idx_to_char = load_vocab(VOCAB_PATH)
    model = load_model(MODEL_PATH, vocab_size=len(char_to_idx))
    training_smiles = load_training_smiles(TRAINING_SMILES_PATH)

    generated = []
    valid_count = 0
    duplicate_valid_count = 0
    seen_valid_smiles = set()
    rows = []

    for _ in range(args.num_samples):
        smiles = sample_smiles(
            model=model,
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
            max_length=args.max_length,
            temperature=args.temperature,
        )
        generated.append(smiles)

        canonical = canonicalize_smiles(smiles)
        if canonical is None:
            continue

        canonical_smiles, mol = canonical
        valid_count += 1
        if canonical_smiles in seen_valid_smiles:
            duplicate_valid_count += 1
            continue

        seen_valid_smiles.add(canonical_smiles)
        rows.append(molecule_row(canonical_smiles, mol, training_smiles))

    rows.sort(key=lambda row: (bool(row["is_novel"]), float(row["qed"])), reverse=True)
    save_generated(rows, GENERATED_CSV_PATH)

    unique_valid_count = len(rows)
    novel_count = sum(1 for row in rows if row["is_novel"])
    validity_percentage = 100.0 * valid_count / max(args.num_samples, 1)
    uniqueness_percentage = 100.0 * unique_valid_count / max(valid_count, 1)
    novelty_percentage = 100.0 * novel_count / max(unique_valid_count, 1)

    summary = {
        "num_samples_requested": args.num_samples,
        "max_length": args.max_length,
        "temperature": args.temperature,
        "generated_count": len(generated),
        "valid_count": valid_count,
        "invalid_count": args.num_samples - valid_count,
        "duplicate_valid_count": duplicate_valid_count,
        "unique_valid_count": unique_valid_count,
        "novel_unique_count": novel_count,
        "validity_percentage": validity_percentage,
        "uniqueness_percentage": uniqueness_percentage,
        "novelty_percentage": novelty_percentage,
        "training_smiles_count": len(training_smiles),
    }
    save_summary(summary, SUMMARY_PATH)

    print(f"generated SMILES: {len(generated)}")
    print(f"valid molecules: {valid_count}")
    print(f"unique valid molecules: {unique_valid_count}")
    print(f"novel unique molecules: {novel_count}")
    print(f"validity percentage: {validity_percentage:.2f}")
    print(f"uniqueness percentage: {uniqueness_percentage:.2f}")
    print(f"novelty percentage: {novelty_percentage:.2f}")
    print("top generated valid molecules:")
    for row in rows[:10]:
        print(
            f"{row['smiles']} | QED={float(row['qed']):.3f} | "
            f"MW={float(row['mol_weight']):.1f} | LogP={float(row['logp']):.2f} | "
            f"novel={row['is_novel']}"
        )
    print(f"saved generated molecules: {GENERATED_CSV_PATH}")
    print(f"saved quality summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
