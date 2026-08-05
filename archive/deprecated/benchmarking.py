import csv
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_SEEDS = [42, 123, 777, 2024, 3407]
METRIC_KEYS = [
    "mse",
    "rmse",
    "mae",
    "median_absolute_error",
    "pcc",
    "spearman",
    "r2",
    "bias",
    "mean_prediction",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return str(value)


def write_json(path: Path, payload: Mapping) -> None:
    def sanitize(value):
        if isinstance(value, Mapping):
            return {str(key): sanitize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        if isinstance(value, tuple):
            return [sanitize(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        if hasattr(value, "item"):
            try:
                return sanitize(value.item())
            except Exception:
                return str(value)
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(sanitize(payload), handle, indent=2, sort_keys=True, default=json_default, allow_nan=False)


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_ids(path: Path) -> List[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip().lower() for line in handle if line.strip()]


def write_ids(path: Path, ids: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for pdb_id in ids:
            handle.write(f"{pdb_id}\n")


def stable_hash(text: str) -> int:
    value = 2166136261
    for char in text:
        value ^= ord(char)
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def next_run_dir(root: Path, prefix: str = "run_") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    max_index = 0
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith(prefix):
            suffix = child.name[len(prefix) :]
            if suffix.isdigit():
                max_index = max(max_index, int(suffix))
    return root / f"{prefix}{max_index + 1:03d}"


def resolve_experiment_dir(
    experiments_root: Path,
    experiment_dir: Optional[str] = None,
    experiment_id: Optional[str] = None,
) -> Path:
    if experiment_dir:
        path = Path(experiment_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    if experiment_id:
        path = experiments_root / experiment_id
        path.mkdir(parents=True, exist_ok=True)
        return path
    path = next_run_dir(experiments_root)
    path.mkdir(parents=True, exist_ok=False)
    return path


def git_info(cwd: Optional[Path] = None) -> Dict[str, object]:
    cwd = cwd or Path.cwd()
    info: Dict[str, object] = {
        "commit": None,
        "branch": None,
        "dirty": None,
        "available": False,
    }
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        ).stdout.strip()
        info.update({"commit": commit, "branch": branch, "dirty": bool(status), "available": True})
    except Exception as exc:
        info["error"] = str(exc)
    return info


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(name)
    except Exception:
        return None


def capture_environment() -> Dict[str, object]:
    env: Dict[str, object] = {
        "timestamp_utc": utc_now(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "hardware": {
            "cpu_count": os.cpu_count(),
            "ram_gb": None,
            "gpu": [],
        },
        "packages": {},
    }
    for package in [
        "torch",
        "torch-geometric",
        "torch-scatter",
        "torch-sparse",
        "numpy",
        "scipy",
        "scikit-learn",
        "rdkit",
        "biopython",
        "matplotlib",
        "tqdm",
    ]:
        env["packages"][package] = _package_version(package)

    try:
        import psutil  # type: ignore

        env["hardware"]["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 3)
    except Exception:
        env["hardware"]["ram_gb"] = None

    try:
        import torch  # type: ignore

        cuda_available = bool(torch.cuda.is_available())
        env["torch_runtime"] = {
            "version": getattr(torch, "__version__", None),
            "cuda_available": cuda_available,
            "cuda_version": getattr(torch.version, "cuda", None),
            "cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
        }
        if cuda_available:
            env["hardware"]["gpu"] = [
                {
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "capability": torch.cuda.get_device_capability(idx),
                    "memory_gb": round(torch.cuda.get_device_properties(idx).total_memory / (1024**3), 3),
                }
                for idx in range(torch.cuda.device_count())
            ]
    except Exception as exc:
        env["torch_runtime"] = {"available": False, "error": str(exc)}

    return env


def args_to_config(args) -> Dict[str, object]:
    payload = vars(args).copy()
    payload["created_utc"] = utc_now()
    return payload


def _fraction_counts(num_items: int, val_fraction: float, test_fraction: float) -> Tuple[int, int, int]:
    if num_items <= 0:
        return 0, 0, 0
    if num_items == 1:
        return 1, 1, 1
    if num_items == 2:
        return 1, 1, 1
    train_count = max(1, int(round((1.0 - val_fraction - test_fraction) * num_items)))
    val_count = max(1, int(round(val_fraction * num_items)))
    test_count = num_items - train_count - val_count
    while test_count < 1 and train_count > 1:
        train_count -= 1
        test_count += 1
    while train_count + val_count + test_count > num_items and train_count > 1:
        train_count -= 1
    while train_count + val_count + test_count < num_items:
        train_count += 1
    return train_count, val_count, test_count


def _split_ordered_ids(ids: Sequence[str], val_fraction: float, test_fraction: float) -> Tuple[List[str], List[str], List[str]]:
    ordered = list(ids)
    num_items = len(ordered)
    if num_items == 0:
        return [], [], []
    if num_items == 1:
        return ordered, ordered, ordered
    if num_items == 2:
        return [ordered[0]], [ordered[0]], [ordered[1]]
    train_count, val_count, _ = _fraction_counts(num_items, val_fraction, test_fraction)
    train_ids = ordered[:train_count]
    val_ids = ordered[train_count : train_count + val_count]
    test_ids = ordered[train_count + val_count :]
    return train_ids, val_ids, test_ids


def deterministic_random_split(
    ids: Sequence[str],
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> Tuple[List[str], List[str], List[str]]:
    shuffled = list(ids)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    return _split_ordered_ids(shuffled, val_fraction, test_fraction)


def _assign_groups_to_splits(
    group_to_ids: Mapping[str, Sequence[str]],
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> Tuple[List[str], List[str], List[str]]:
    groups = [(group, sorted(ids)) for group, ids in group_to_ids.items()]
    rng = random.Random(seed)
    rng.shuffle(groups)

    total = sum(len(ids) for _, ids in groups)
    if total <= 2:
        flat = [pdb_id for _, ids in groups for pdb_id in ids]
        return _split_ordered_ids(flat, val_fraction, test_fraction)

    train_target, val_target, test_target = _fraction_counts(total, val_fraction, test_fraction)
    splits = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}
    targets = {"train": train_target, "val": val_target, "test": test_target}

    for _, ids in sorted(groups, key=lambda item: len(item[1]), reverse=True):
        deficits = {name: targets[name] - counts[name] for name in splits}
        destination = max(deficits, key=lambda name: (deficits[name], -counts[name]))
        splits[destination].extend(ids)
        counts[destination] += len(ids)

    for name in splits:
        splits[name] = sorted(splits[name])
    if not splits["val"]:
        splits["val"] = splits["train"][:1]
    if not splits["test"]:
        splits["test"] = splits["train"][-1:]
    return splits["train"], splits["val"], splits["test"]


def _parse_pdbbind_years(data_dir: Path) -> Dict[str, int]:
    index_path = data_dir / "index" / "INDEX_general_PL.2020R1.lst"
    years: Dict[str, int] = {}
    if not index_path.exists():
        return years
    with index_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3 and parts[2].isdigit():
                years[parts[0].lower()] = int(parts[2])
    return years


def _protein_family_groups(ids: Sequence[str], metadata_by_id: Mapping[str, Mapping]) -> Tuple[Dict[str, List[str]], List[str]]:
    warnings: List[str] = []
    groups: Dict[str, List[str]] = {}
    missing = 0
    for pdb_id in ids:
        metadata = metadata_by_id.get(pdb_id, {})
        family = (
            metadata.get("protein_family")
            or metadata.get("family")
            or metadata.get("uniprot_family")
            or metadata.get("uniprot_id")
        )
        if not family:
            missing += 1
            family = f"pdb_prefix:{pdb_id[:2]}"
        groups.setdefault(str(family), []).append(pdb_id)
    if missing:
        warnings.append(
            "Protein-family split requested, but explicit family metadata was missing for "
            f"{missing} complexes; deterministic PDB-prefix groups were used for those entries."
        )
    return groups, warnings


def _scaffold_groups(ids: Sequence[str], metadata_by_id: Mapping[str, Mapping]) -> Tuple[Dict[str, List[str]], List[str]]:
    warnings: List[str] = []
    groups: Dict[str, List[str]] = {}
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem.Scaffolds import MurckoScaffold  # type: ignore
    except Exception as exc:
        warnings.append(f"RDKit scaffold extraction unavailable ({exc}); deterministic PDB-ID groups were used.")
        return {f"id:{pdb_id}": [pdb_id] for pdb_id in ids}, warnings

    missing = 0
    for pdb_id in ids:
        metadata = metadata_by_id.get(pdb_id, {})
        ligand_path = metadata.get("ligand_path")
        scaffold = None
        if ligand_path:
            path = Path(str(ligand_path))
            try:
                if path.suffix.lower() == ".sdf":
                    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
                    mol = supplier[0] if supplier and len(supplier) else None
                elif path.suffix.lower() == ".mol2":
                    mol = Chem.MolFromMol2File(str(path), removeHs=False)
                else:
                    mol = None
                if mol is not None:
                    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            except Exception:
                scaffold = None
        if not scaffold:
            missing += 1
            scaffold = f"id:{pdb_id}"
        groups.setdefault(scaffold, []).append(pdb_id)
    if missing:
        warnings.append(
            "Scaffold split requested, but ligand scaffold extraction failed or was unavailable for "
            f"{missing} complexes; those entries were treated as singleton scaffold groups."
        )
    return groups, warnings


def build_split_manifest(
    ids: Sequence[str],
    seed: int,
    strategy: str,
    split_dir: Path,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    fixed_split_dir: Optional[Path] = None,
    metadata_by_id: Optional[Mapping[str, Mapping]] = None,
    data_dir: Optional[Path] = None,
) -> Dict[str, object]:
    ids = [pdb_id.lower() for pdb_id in ids]
    metadata_by_id = metadata_by_id or {}
    split_dir.mkdir(parents=True, exist_ok=True)
    warnings: List[str] = []
    effective_strategy = strategy

    if strategy == "fixed":
        source = fixed_split_dir or split_dir
        train_ids = read_ids(source / "train_ids.txt")
        val_ids = read_ids(source / "val_ids.txt")
        test_ids = read_ids(source / "test_ids.txt")
        if not train_ids or not val_ids or not test_ids:
            raise ValueError(
                "Fixed split requires non-empty train_ids.txt, val_ids.txt, and test_ids.txt "
                f"under {source}."
            )
        missing = sorted((set(train_ids) | set(val_ids) | set(test_ids)) - set(ids))
        if missing:
            raise ValueError(f"Fixed split contains IDs not present in the processed dataset: {missing[:10]}")
    elif strategy == "temporal":
        years = _parse_pdbbind_years(data_dir) if data_dir else {}
        available = [pdb_id for pdb_id in ids if pdb_id in years]
        if len(available) == len(ids) and len(ids) > 2:
            ordered = sorted(ids, key=lambda pdb_id: (years[pdb_id], pdb_id))
            train_ids, val_ids, test_ids = _split_ordered_ids(ordered, val_fraction, test_fraction)
        else:
            warnings.append(
                "Temporal split requested, but complete release-year metadata was not available; "
                "falling back to deterministic seed split."
            )
            effective_strategy = "random_fallback_for_temporal"
            train_ids, val_ids, test_ids = deterministic_random_split(ids, seed, val_fraction, test_fraction)
    elif strategy == "scaffold":
        groups, scaffold_warnings = _scaffold_groups(ids, metadata_by_id)
        warnings.extend(scaffold_warnings)
        train_ids, val_ids, test_ids = _assign_groups_to_splits(groups, seed, val_fraction, test_fraction)
    elif strategy == "protein_family":
        groups, family_warnings = _protein_family_groups(ids, metadata_by_id)
        warnings.extend(family_warnings)
        train_ids, val_ids, test_ids = _assign_groups_to_splits(groups, seed, val_fraction, test_fraction)
    else:
        effective_strategy = "random"
        train_ids, val_ids, test_ids = deterministic_random_split(ids, seed, val_fraction, test_fraction)

    write_ids(split_dir / "train_ids.txt", train_ids)
    write_ids(split_dir / "val_ids.txt", val_ids)
    write_ids(split_dir / "test_ids.txt", test_ids)

    overlap = {
        "train_val": sorted(set(train_ids) & set(val_ids)),
        "train_test": sorted(set(train_ids) & set(test_ids)),
        "val_test": sorted(set(val_ids) & set(test_ids)),
    }
    if len(ids) > 2 and any(overlap.values()):
        warnings.append("Split overlap detected; this should only occur for tiny smoke-test datasets.")

    manifest = {
        "created_utc": utc_now(),
        "strategy": strategy,
        "effective_strategy": effective_strategy,
        "seed": seed,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "num_total": len(ids),
        "counts": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
        "ids": {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        },
        "overlap": overlap,
        "warnings": warnings,
        "split_files": {
            "train": str(split_dir / "train_ids.txt"),
            "val": str(split_dir / "val_ids.txt"),
            "test": str(split_dir / "test_ids.txt"),
            "metadata": str(split_dir / "metadata.json"),
        },
    }
    write_json(split_dir / "metadata.json", manifest)
    return manifest


def split_ids_to_indices(pdb_ids: Sequence[str], split_manifest: Mapping[str, object]) -> Tuple[List[int], List[int], List[int]]:
    index_by_id = {pdb_id.lower(): idx for idx, pdb_id in enumerate(pdb_ids)}
    ids = split_manifest.get("ids", {})
    if not isinstance(ids, Mapping):
        raise ValueError("Split manifest is missing ids mapping.")

    def convert(split_name: str) -> List[int]:
        return [index_by_id[pdb_id] for pdb_id in ids.get(split_name, []) if pdb_id in index_by_id]

    return convert("train"), convert("val"), convert("test")


def copy_split_manifest(split_dir: Path, experiment_dir: Path) -> Path:
    target = experiment_dir / "split"
    target.mkdir(parents=True, exist_ok=True)
    for name in ["train_ids.txt", "val_ids.txt", "test_ids.txt", "metadata.json"]:
        source = split_dir / name
        if source.exists():
            shutil.copy2(source, target / name)
    return target


def _as_float_list(values: Sequence[object]) -> List[float]:
    result = []
    for value in values:
        try:
            number = float(value)
        except Exception:
            continue
        if not math.isnan(number):
            result.append(number)
    return result


def _rankdata(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    centered_x = [value - mean_x for value in x]
    centered_y = [value - mean_y for value in y]
    denom = math.sqrt(sum(value * value for value in centered_x) * sum(value * value for value in centered_y))
    if denom == 0:
        return float("nan")
    return sum(a * b for a, b in zip(centered_x, centered_y)) / denom


def compute_regression_metrics(y_true: Sequence[object], y_pred: Sequence[object]) -> Dict[str, float]:
    true_values = _as_float_list(y_true)
    pred_values = _as_float_list(y_pred)
    pairs = [(true, pred) for true, pred in zip(true_values, pred_values)]
    if not pairs:
        return {key: float("nan") for key in METRIC_KEYS}

    true_values = [true for true, _ in pairs]
    pred_values = [pred for _, pred in pairs]
    errors = [pred - true for true, pred in pairs]
    abs_errors = [abs(error) for error in errors]
    squared_errors = [error * error for error in errors]
    mse = statistics.fmean(squared_errors)
    rmse = math.sqrt(mse)
    mae = statistics.fmean(abs_errors)
    median_ae = statistics.median(abs_errors)
    true_mean = statistics.fmean(true_values)
    ss_res = sum(squared_errors)
    ss_tot = sum((true - true_mean) ** 2 for true in true_values)
    r2 = float("nan") if ss_tot == 0 else 1.0 - (ss_res / ss_tot)
    pcc = _pearson(true_values, pred_values)
    spearman = _pearson(_rankdata(true_values), _rankdata(pred_values)) if len(true_values) > 1 else float("nan")

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "median_absolute_error": float(median_ae),
        "pcc": float(pcc),
        "spearman": float(spearman),
        "r2": float(r2),
        "bias": float(statistics.fmean(errors)),
        "mean_prediction": float(statistics.fmean(pred_values)),
    }


def save_predictions_csv(
    pdb_ids: Sequence[str],
    y_true: Sequence[object],
    y_pred: Sequence[object],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pdb_id", "true_affinity", "predicted_affinity", "residual", "absolute_error", "squared_error"])
        for pdb_id, true_value, pred_value in zip(pdb_ids, y_true, y_pred):
            true_float = float(true_value)
            pred_float = float(pred_value)
            residual = pred_float - true_float
            writer.writerow([pdb_id, true_float, pred_float, residual, abs(residual), residual * residual])


def build_error_rows(pdb_ids: Sequence[str], y_true: Sequence[object], y_pred: Sequence[object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for pdb_id, true_value, pred_value in zip(pdb_ids, y_true, y_pred):
        true_float = float(true_value)
        pred_float = float(pred_value)
        residual = pred_float - true_float
        rows.append(
            {
                "pdb_id": pdb_id,
                "true_affinity": true_float,
                "predicted_affinity": pred_float,
                "residual": residual,
                "absolute_error": abs(residual),
                "squared_error": residual * residual,
            }
        )
    ranked = sorted(rows, key=lambda row: row["absolute_error"])
    for rank, row in enumerate(ranked, start=1):
        row["rank_best"] = rank
    for rank, row in enumerate(reversed(ranked), start=1):
        row["rank_worst"] = rank
    return sorted(rows, key=lambda row: row["absolute_error"], reverse=True)


def save_error_analysis(
    pdb_ids: Sequence[str],
    y_true: Sequence[object],
    y_pred: Sequence[object],
    csv_path: Path,
    report_path: Path,
) -> Dict[str, object]:
    rows = build_error_rows(pdb_ids, y_true, y_pred)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pdb_id",
        "true_affinity",
        "predicted_affinity",
        "residual",
        "absolute_error",
        "squared_error",
        "rank_best",
        "rank_worst",
        "category",
    ]
    abs_errors = [float(row["absolute_error"]) for row in rows]
    mean_abs = statistics.fmean(abs_errors) if abs_errors else float("nan")
    std_abs = statistics.stdev(abs_errors) if len(abs_errors) > 1 else 0.0
    threshold = mean_abs + 2.0 * std_abs
    best = sorted(rows, key=lambda row: row["absolute_error"])[:20]
    worst = rows[:20]
    outliers = [row for row in rows if float(row["absolute_error"]) >= threshold] if rows else []
    bias = statistics.fmean(float(row["residual"]) for row in rows) if rows else float("nan")
    mean_pred = statistics.fmean(float(row["predicted_affinity"]) for row in rows) if rows else float("nan")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            category = "typical"
            if row in worst:
                category = "largest_error"
            if row in best:
                category = "best_prediction"
            if row in outliers:
                category = "outlier"
            output = dict(row)
            output["category"] = category
            writer.writerow(output)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# Error Analysis\n\n")
        handle.write(f"- Sample count: {len(rows)}\n")
        handle.write(f"- Bias (mean residual): {bias:.6f}\n")
        handle.write(f"- Average prediction: {mean_pred:.6f}\n")
        handle.write(f"- Outlier threshold: absolute_error >= {threshold:.6f}\n")
        handle.write(f"- Outlier count: {len(outliers)}\n\n")
        handle.write("## Top 20 Largest Errors\n\n")
        handle.write("| Rank | PDB ID | True | Predicted | Residual | Absolute Error |\n")
        handle.write("| --- | --- | ---: | ---: | ---: | ---: |\n")
        for idx, row in enumerate(worst, start=1):
            handle.write(
                f"| {idx} | {row['pdb_id']} | {row['true_affinity']:.6f} | "
                f"{row['predicted_affinity']:.6f} | {row['residual']:.6f} | {row['absolute_error']:.6f} |\n"
            )
        handle.write("\n## Top 20 Best Predictions\n\n")
        handle.write("| Rank | PDB ID | True | Predicted | Residual | Absolute Error |\n")
        handle.write("| --- | --- | ---: | ---: | ---: | ---: |\n")
        for idx, row in enumerate(best, start=1):
            handle.write(
                f"| {idx} | {row['pdb_id']} | {row['true_affinity']:.6f} | "
                f"{row['predicted_affinity']:.6f} | {row['residual']:.6f} | {row['absolute_error']:.6f} |\n"
            )
    return {
        "sample_count": len(rows),
        "bias": bias,
        "average_prediction": mean_pred,
        "outlier_threshold": threshold,
        "outlier_count": len(outliers),
        "top_error_pdb_ids": [row["pdb_id"] for row in worst],
        "best_prediction_pdb_ids": [row["pdb_id"] for row in best],
    }


def _calibration_bins(y_true: Sequence[object], y_pred: Sequence[object], num_bins: int = 10) -> List[Dict[str, float]]:
    pairs = sorted((float(pred), float(true)) for true, pred in zip(y_true, y_pred))
    if not pairs:
        return []
    bin_count = min(num_bins, len(pairs))
    bins: List[Dict[str, float]] = []
    for idx in range(bin_count):
        start = int(idx * len(pairs) / bin_count)
        end = int((idx + 1) * len(pairs) / bin_count)
        chunk = pairs[start:end]
        if not chunk:
            continue
        pred_values = [pred for pred, _ in chunk]
        true_values = [true for _, true in chunk]
        bins.append(
            {
                "bin": idx + 1,
                "count": len(chunk),
                "mean_prediction": statistics.fmean(pred_values),
                "mean_true": statistics.fmean(true_values),
                "calibration_error": statistics.fmean(pred_values) - statistics.fmean(true_values),
            }
        )
    return bins


def save_calibration_csv(y_true: Sequence[object], y_pred: Sequence[object], output_path: Path) -> List[Dict[str, float]]:
    bins = _calibration_bins(y_true, y_pred)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["bin", "count", "mean_prediction", "mean_true", "calibration_error"],
        )
        writer.writeheader()
        writer.writerows(bins)
    return bins


def save_evaluation_plots(
    y_true: Sequence[object],
    y_pred: Sequence[object],
    output_dir: Path,
    run_name: str,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: Dict[str, str] = {}
    true_values = [float(value) for value in y_true]
    pred_values = [float(value) for value in y_pred]
    residuals = [pred - true for true, pred in zip(true_values, pred_values)]
    abs_errors = [abs(error) for error in residuals]
    calibration = _calibration_bins(true_values, pred_values)
    save_calibration_csv(true_values, pred_values, output_dir / "calibration_curve.csv")

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        warning_path = output_dir / "plot_warnings.txt"
        warning_path.write_text(f"Matplotlib unavailable; plots were not generated: {exc}\n", encoding="utf-8")
        artifacts["warnings"] = str(warning_path)
        return artifacts

    if not true_values:
        warning_path = output_dir / "plot_warnings.txt"
        warning_path.write_text("No predictions were available for plotting.\n", encoding="utf-8")
        artifacts["warnings"] = str(warning_path)
        return artifacts

    min_val = min(min(true_values), min(pred_values))
    max_val = max(max(true_values), max(pred_values))

    def save_current(name: str) -> None:
        path = output_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()
        artifacts[name.replace(".png", "")] = str(path)

    plt.figure(figsize=(6, 6))
    plt.scatter(true_values, pred_values, alpha=0.7, edgecolors="none")
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=1.0, color="tab:red")
    plt.xlabel("True affinity")
    plt.ylabel("Predicted affinity")
    plt.title(f"Prediction Scatter: {run_name}")
    save_current("prediction_scatter.png")

    plt.figure(figsize=(7, 5))
    plt.scatter(true_values, residuals, alpha=0.7, edgecolors="none")
    plt.axhline(0.0, linestyle="--", linewidth=1.0, color="tab:red")
    plt.xlabel("True affinity")
    plt.ylabel("Residual (predicted - true)")
    plt.title(f"Residual Scatter: {run_name}")
    save_current("residual_scatter.png")

    plt.figure(figsize=(7, 5))
    plt.hist(abs_errors, bins=min(30, max(5, len(abs_errors))), alpha=0.85)
    plt.xlabel("Absolute error")
    plt.ylabel("Count")
    plt.title(f"Error Histogram: {run_name}")
    save_current("error_histogram.png")

    plt.figure(figsize=(7, 5))
    plt.hist(residuals, bins=min(30, max(5, len(residuals))), alpha=0.85)
    plt.axvline(0.0, linestyle="--", linewidth=1.0, color="tab:red")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title(f"Residual Distribution: {run_name}")
    save_current("residual_distribution.png")

    plt.figure(figsize=(7, 5))
    plt.hist(pred_values, bins=min(30, max(5, len(pred_values))), alpha=0.85)
    plt.xlabel("Predicted affinity")
    plt.ylabel("Count")
    plt.title(f"Prediction Histogram: {run_name}")
    save_current("prediction_histogram.png")

    if calibration:
        plt.figure(figsize=(6, 6))
        xs = [row["mean_true"] for row in calibration]
        ys = [row["mean_prediction"] for row in calibration]
        cal_min = min(min(xs), min(ys))
        cal_max = max(max(xs), max(ys))
        plt.plot([cal_min, cal_max], [cal_min, cal_max], linestyle="--", linewidth=1.0, color="tab:red")
        plt.plot(xs, ys, marker="o")
        plt.xlabel("Mean true affinity")
        plt.ylabel("Mean predicted affinity")
        plt.title(f"Calibration Curve: {run_name}")
        save_current("calibration_curve.png")

    artifacts["calibration_curve_csv"] = str(output_dir / "calibration_curve.csv")
    return artifacts


def save_evaluation_report(metrics: Mapping[str, float], output_path: Path, title: str = "Evaluation Report") -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# {title}\n\n")
        handle.write("| Metric | Value |\n")
        handle.write("| --- | ---: |\n")
        for key in METRIC_KEYS:
            value = metrics.get(key, float("nan"))
            if isinstance(value, float) and math.isnan(value):
                display = "nan"
            else:
                display = f"{float(value):.6f}"
            handle.write(f"| {key} | {display} |\n")


def _ci95(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = statistics.fmean(values)
    if len(values) == 1:
        return mean, mean
    std = statistics.stdev(values)
    margin = 1.96 * std / math.sqrt(len(values))
    return mean - margin, mean + margin


def _bootstrap_ci(values: Sequence[float], seed: int = 3407, rounds: int = 2000) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means = []
    for _ in range(rounds):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(statistics.fmean(sample))
    means.sort()
    lower_index = int(0.025 * (len(means) - 1))
    upper_index = int(0.975 * (len(means) - 1))
    return means[lower_index], means[upper_index]


def _format_float(value: object) -> str:
    try:
        number = float(value)
    except Exception:
        return "nan"
    if math.isnan(number):
        return "nan"
    return f"{number:.6f}"


def aggregate_metrics(seed_metrics: Sequence[Mapping[str, object]]) -> Dict[str, Dict[str, float]]:
    aggregate: Dict[str, Dict[str, float]] = {}
    for metric in METRIC_KEYS:
        values = []
        for row in seed_metrics:
            try:
                value = float(row.get(metric, float("nan")))
            except Exception:
                continue
            if not math.isnan(value):
                values.append(value)
        if not values:
            continue
        ci_low, ci_high = _ci95(values)
        boot_low, boot_high = _bootstrap_ci(values)
        aggregate[metric] = {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "bootstrap_ci95_low": boot_low,
            "bootstrap_ci95_high": boot_high,
            "n": float(len(values)),
        }
    return aggregate


def write_results_summary(
    seed_rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    benchmark_name: str = "ProtLigGNN Benchmark",
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate_metrics(seed_rows)
    csv_path = output_dir / "results_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "metric",
            "mean",
            "median",
            "std",
            "min",
            "max",
            "ci95_low",
            "ci95_high",
            "bootstrap_ci95_low",
            "bootstrap_ci95_high",
            "n",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metric, values in summary.items():
            writer.writerow({"metric": metric, **values})

    per_seed_path = output_dir / "per_seed_metrics.csv"
    per_seed_keys = ["seed", "run_dir"] + METRIC_KEYS
    with per_seed_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_seed_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(seed_rows)

    report_path = output_dir / "benchmark_report.md"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# {benchmark_name}\n\n")
        handle.write(f"Generated UTC: {utc_now()}\n\n")
        handle.write("## Aggregate Metrics\n\n")
        handle.write("| Metric | Mean +/- Std | CI95 | Bootstrap CI95 | Best | Worst | N |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for metric, values in summary.items():
            handle.write(
                f"| {metric} | {values['mean']:.6f} +/- {values['std']:.6f} | "
                f"[{values['ci95_low']:.6f}, {values['ci95_high']:.6f}] | "
                f"[{values['bootstrap_ci95_low']:.6f}, {values['bootstrap_ci95_high']:.6f}] | "
                f"{values['min']:.6f} | {values['max']:.6f} | {int(values['n'])} |\n"
            )
        handle.write("\n## Paired Statistical Tests\n\n")
        handle.write(
            "No paired baseline metrics were supplied to this benchmark runner, so paired tests are not applicable. "
            "The aggregation code preserves per-seed rows so future baseline-vs-candidate paired tests can be added "
            "without changing the experiment artifact format.\n"
        )
        handle.write("\n## Per-Seed Runs\n\n")
        handle.write("| Seed | Run Dir | RMSE | MAE | Pearson | Spearman | R2 |\n")
        handle.write("| ---: | --- | ---: | ---: | ---: | ---: | ---: |\n")
        for row in seed_rows:
            handle.write(
                f"| {row.get('seed')} | {row.get('run_dir')} | "
                f"{_format_float(row.get('rmse'))} | {_format_float(row.get('mae'))} | "
                f"{_format_float(row.get('pcc'))} | {_format_float(row.get('spearman'))} | "
                f"{_format_float(row.get('r2'))} |\n"
            )

    write_json(output_dir / "benchmark_summary.json", {"summary": summary, "runs": list(seed_rows)})
    return {
        "summary": summary,
        "results_summary_csv": str(csv_path),
        "per_seed_metrics_csv": str(per_seed_path),
        "benchmark_report": str(report_path),
    }


def build_artifact_manifest(
    experiment_dir: Path,
    config_path: Path,
    environment_path: Path,
    split_dir: Path,
    metrics_path: Path,
    predictions_path: Path,
    plots_dir: Path,
    checkpoint_path: Path,
    history_path: Path,
    log_path: Path,
    extra_artifacts: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    manifest = {
        "created_utc": utc_now(),
        "experiment_dir": str(experiment_dir),
        "config": str(config_path),
        "environment": str(environment_path),
        "split": {
            "directory": str(split_dir),
            "train_ids": str(split_dir / "train_ids.txt"),
            "val_ids": str(split_dir / "val_ids.txt"),
            "test_ids": str(split_dir / "test_ids.txt"),
            "metadata": str(split_dir / "metadata.json"),
        },
        "metrics": str(metrics_path),
        "predictions": str(predictions_path),
        "plots": str(plots_dir),
        "checkpoint": str(checkpoint_path),
        "training_history": str(history_path),
        "logs": str(log_path),
        "extra_artifacts": dict(extra_artifacts or {}),
    }
    write_json(experiment_dir / "manifest.json", manifest)
    return manifest
