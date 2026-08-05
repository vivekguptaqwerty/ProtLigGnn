from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import protliggnn_train as train


def parse_boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def checkpoint_args(checkpoint: dict[str, Any]) -> SimpleNamespace:
    defaults = vars(train.parse_args([]))
    saved_args = checkpoint.get("args", {})
    if isinstance(saved_args, argparse.Namespace):
        saved_args = vars(saved_args)
    defaults.update(saved_args)

    for key in [
        "attention_learnable_scale",
        "attention_normalize_bias",
        "no_attention",
        "no_crossgraph",
        "no_layernorm",
        "no_residual",
    ]:
        defaults[key] = parse_boolish(defaults.get(key, False))

    defaults.setdefault("model_type", "protliggnn")
    defaults.setdefault("pooling", "mean")
    defaults.setdefault("hidden_dim", 256)
    defaults.setdefault("dropout", 0.2)
    defaults.setdefault("contact_hidden_dim", 256)
    return SimpleNamespace(**defaults)


def read_coreset(path: Path, limit: int | None = None) -> list[tuple[str, float, int]]:
    rows: list[tuple[str, float, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 6:
            continue
        rows.append((parts[0].lower(), float(parts[3]), int(parts[5])))
        if limit is not None and len(rows) >= limit:
            break
    return rows


def make_records(casf_dir: Path, coreset_path: Path, limit: int | None) -> list[train.ComplexRecord]:
    records: list[train.ComplexRecord] = []
    for pdb_id, affinity, _target in read_coreset(coreset_path, limit=limit):
        complex_dir = casf_dir / "coreset" / pdb_id
        protein_path = complex_dir / f"{pdb_id}_protein.pdb"
        ligand_path = train.select_ligand_file(complex_dir, pdb_id)
        if not protein_path.exists() or ligand_path is None:
            print(f"Skipping {pdb_id}: missing protein or ligand file")
            continue
        records.append(
            train.ComplexRecord(
                pdb_id=pdb_id,
                affinity=affinity,
                protein_path=protein_path,
                ligand_path=ligand_path,
            )
        )
    return records


def build_model(args, device: torch.device) -> torch.nn.Module:
    common_kwargs = {
        "ligand_dim": train.LIGAND_FEATURE_DIM,
        "protein_dim": train.PROTEIN_FEATURE_DIM,
        "hidden_dim": getattr(args, "hidden_dim", 256),
        "no_crossgraph": getattr(args, "no_crossgraph", False),
        "dropout": getattr(args, "dropout", 0.2),
        "use_residual": not getattr(args, "no_residual", False),
        "use_layernorm": not getattr(args, "no_layernorm", False),
        "use_attention": not getattr(args, "no_attention", False),
        "pooling": getattr(args, "pooling", "mean"),
    }
    model_type = getattr(args, "model_type", "protliggnn")
    if model_type == "soft_routing":
        from models.soft_routing import ProtLigGNNSoftRouting, RoutingConfig
        config = RoutingConfig(
            router_type=getattr(args, "router_type", "gated"),
            routing_level=getattr(args, "routing_level", "embedding"),
            latent_dimension=common_kwargs["hidden_dim"] * 2,
            routing_dimension=common_kwargs["hidden_dim"],
            num_layers=getattr(args, "router_layers", 2),
            activation=getattr(args, "router_activation", "relu"),
            normalization=getattr(args, "router_normalization", "layer"),
            dropout=common_kwargs["dropout"] * 0.5,
            router_capacity=getattr(args, "router_capacity", "base"),
        )
        return ProtLigGNNSoftRouting(**common_kwargs, config=config).to(device)
    elif model_type == "geometry_attention":
        from models.geometry_attention import AttentionBiasConfig, ProtLigGNNGeometryAttention
        config = AttentionBiasConfig(
            bias_type=getattr(args, "attention_bias_type", "rbf"),
            num_rbf=getattr(args, "rbf_basis", 32),
            cutoff_distance=getattr(args, "rbf_stop", 12.0),
            learnable_scale=parse_boolish(getattr(args, "attention_learnable_scale", True)),
            initial_alpha=getattr(args, "attention_initial_alpha", 0.1),
            normalize_bias=parse_boolish(getattr(args, "attention_normalize_bias", True)),
            num_heads=getattr(args, "attention_heads", 4),
        )
        return ProtLigGNNGeometryAttention(**common_kwargs, config=config).to(device)
    else:
        return train.ProtLigGNN(**common_kwargs).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CASF-2016 coreset predictions from a ProtLigGNN checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--casf_dir", default=str(ROOT / "data" / "CASF-2016"))
    parser.add_argument("--coreset", default=str(ROOT / "data" / "CASF-2016" / "power_scoring" / "CoreSet.dat"))
    parser.add_argument("--output", default=str(ROOT / "scratch" / "casf2016_predictions.csv"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_args = checkpoint_args(checkpoint)
    model = build_model(model_args, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()


    coreset_rows = {code: (affinity, target) for code, affinity, target in read_coreset(Path(args.coreset))}
    records = make_records(Path(args.casf_dir), Path(args.coreset), args.limit)
    dataset = train.PDBbindPairDataset(records)
    loader = DataLoader(dataset.samples, batch_size=args.batch_size, shuffle=False, collate_fn=train.collate_pairs)

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for ligand_batch, protein_batch, labels, pdb_ids in loader:
            ligand_batch = ligand_batch.to(device)
            protein_batch = protein_batch.to(device)
            model_output = model(ligand_batch, protein_batch)
            if isinstance(model_output, tuple):
                preds = model_output[0]
            else:
                preds = model_output
            preds_np = preds.detach().cpu().numpy()
            labels_np = labels.numpy()
            for pdb_id, true_value, pred_value in zip(pdb_ids, labels_np, preds_np):
                affinity, target = coreset_rows[pdb_id]
                absolute_error = abs(float(pred_value) - float(true_value))
                rows.append(
                    {
                        "pdb_id": pdb_id,
                        "experimental_affinity": float(affinity),
                        "predicted_affinity": float(pred_value),
                        "absolute_error": absolute_error,
                        "squared_error": absolute_error ** 2,
                        "target": int(target),
                    }
                )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pdb_id",
                "experimental_affinity",
                "predicted_affinity",
                "absolute_error",
                "squared_error",
                "target",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    y_true = np.asarray([row["experimental_affinity"] for row in rows], dtype=np.float32)
    y_pred = np.asarray([row["predicted_affinity"] for row in rows], dtype=np.float32)
    print(f"Wrote {len(rows)} CASF predictions to {output_path}")
    if len(rows) >= 2:
        print(f"Prediction mean={float(y_pred.mean()):.4f}, target mean={float(y_true.mean()):.4f}")


if __name__ == "__main__":
    main()
