"""Research infrastructure utilities for ProtLigGNN experiments.

This module intentionally avoids importing Torch so registry, comparison,
fingerprinting, and validation commands can run in lightweight environments.
"""

import csv
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from collections import Counter
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from benchmarking import capture_environment, read_json, utc_now, write_json


REGISTRY_FIELDS = [
    "Run ID",
    "Timestamp",
    "Git Commit",
    "Dataset Version",
    "Dataset Hash",
    "Config Hash",
    "Split Strategy",
    "Seed",
    "Best Validation RMSE",
    "Test Metrics",
    "Experiment Status",
    "Duration",
    "Checkpoint Path",
    "model_type",
    "bias_type",
    "num_rbf",
    "alpha_init",
    "alpha_final",
    "localization_score",
    "attention_entropy",
    "runtime_overhead",
    "memory_overhead",
]

CONFIG_HASH_EXCLUDE = {
    "allow_duplicate",
    "author",
    "current_milestone",
    "debug",
    "description",
    "disable_experiment_tracking",
    "experiment_dir",
    "experiment_id",
    "experiment_purpose",
    "experiments_root",
    "expected_outcome",
    "log_level",
    "notes",
    "research_question",
    "resume",
    "run_name",
    "save_path",
    "tags",
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Mapping) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def canonical_config(config: Mapping[str, object]) -> Dict[str, object]:
    """Return the experiment-defining subset of a CLI/config mapping."""
    canonical: Dict[str, object] = {}
    for key, value in sorted(config.items()):
        if key in CONFIG_HASH_EXCLUDE:
            continue
        if isinstance(value, Path):
            canonical[key] = str(value)
        else:
            canonical[key] = value
    return canonical


def write_config_hash(config: Mapping[str, object], experiment_dir: Path) -> Dict[str, object]:
    canonical = canonical_config(config)
    digest = hashlib.sha256(canonical_json_bytes(canonical)).hexdigest()
    hash_path = experiment_dir / "config_hash.txt"
    hash_path.write_text(digest + "\n", encoding="utf-8")
    write_json(
        experiment_dir / "config_fingerprint.json",
        {
            "created_utc": utc_now(),
            "sha256": digest,
            "canonical_config": canonical,
            "excluded_keys": sorted(CONFIG_HASH_EXCLUDE),
        },
    )
    return {"sha256": digest, "path": str(hash_path), "canonical_config": canonical}


def _modified_time_utc(path: Path) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(path.stat().st_mtime))


def _strip_volatile_json(value):
    if isinstance(value, Mapping):
        return {
            key: _strip_volatile_json(item)
            for key, item in value.items()
            if key not in {"created_utc", "timestamp_utc", "modified_time_utc", "absolute_path"}
        }
    if isinstance(value, list):
        return [_strip_volatile_json(item) for item in value]
    return value


def stable_file_sha256(path: Path) -> str:
    if path.suffix.lower() == ".json":
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return hashlib.sha256(canonical_json_bytes(_strip_volatile_json(payload))).hexdigest()
        except Exception:
            return sha256_file(path)
    return sha256_file(path)


def _file_record(path: Path, category: str, root: Optional[Path] = None) -> Dict[str, object]:
    resolved = path.resolve()
    if root:
        try:
            display_path = str(resolved.relative_to(root.resolve()))
        except ValueError:
            display_path = str(resolved)
    else:
        display_path = str(resolved)
    stat = resolved.stat()
    return {
        "path": display_path,
        "absolute_path": str(resolved),
        "category": category,
        "sha256": sha256_file(resolved),
        "stable_sha256": stable_file_sha256(resolved),
        "size_bytes": stat.st_size,
        "modified_time_utc": _modified_time_utc(resolved),
    }


def _existing_files(paths: Iterable[Path]) -> List[Path]:
    return [path for path in paths if path.exists() and path.is_file()]


def compute_dataset_fingerprint(
    data_dir: Path,
    split_dir: Path,
    output_path: Path,
    metadata_paths: Optional[Sequence[Path]] = None,
) -> Dict[str, object]:
    """Hash dataset index, split, metadata, and label files for an experiment."""
    metadata_paths = metadata_paths or []
    records: List[Dict[str, object]] = []
    index_dir = data_dir / "index"
    if index_dir.exists():
        for path in sorted(index_dir.iterdir()):
            if path.is_file():
                category = "label_file" if path.name.startswith("INDEX_") else "dataset_index_metadata"
                records.append(_file_record(path, category, root=data_dir))
    for name in ["train_ids.txt", "val_ids.txt", "test_ids.txt", "metadata.json"]:
        path = split_dir / name
        if path.exists():
            records.append(_file_record(path, "split_manifest", root=split_dir.parent))
    for path in _existing_files(metadata_paths):
        records.append(_file_record(path, "experiment_metadata", root=output_path.parent))

    overall_payload = [
        {
            "path": record["path"],
            "category": record["category"],
            "stable_sha256": record["stable_sha256"],
            "size_bytes": record["size_bytes"],
        }
        for record in sorted(records, key=lambda item: (str(item["category"]), str(item["path"])))
    ]
    overall_sha256 = hashlib.sha256(canonical_json_bytes({"files": overall_payload})).hexdigest()
    fingerprint = {
        "created_utc": utc_now(),
        "dataset_root": str(data_dir),
        "split_dir": str(split_dir),
        "overall_sha256": overall_sha256,
        "file_count": len(records),
        "files": records,
    }
    write_json(output_path, fingerprint)
    return fingerprint


def validate_split_manifest(
    split_manifest: Mapping[str, object],
    expected_ids: Sequence[str],
    output_path: Path,
    strict: bool = True,
) -> Dict[str, object]:
    """Validate split disjointness, duplicates, missing IDs, and orphan IDs."""
    ids = split_manifest.get("ids", {})
    if not isinstance(ids, Mapping):
        raise ValueError("Split manifest does not contain an ids mapping.")
    splits = {
        "train": [str(value).lower() for value in ids.get("train", [])],
        "val": [str(value).lower() for value in ids.get("val", [])],
        "test": [str(value).lower() for value in ids.get("test", [])],
    }
    expected = {pdb_id.lower() for pdb_id in expected_ids}
    duplicate_ids = {
        name: sorted([pdb_id for pdb_id, count in Counter(values).items() if count > 1])
        for name, values in splits.items()
    }
    overlap = {
        "train_val": sorted(set(splits["train"]) & set(splits["val"])),
        "train_test": sorted(set(splits["train"]) & set(splits["test"])),
        "val_test": sorted(set(splits["val"]) & set(splits["test"])),
    }
    all_split_ids = set().union(*(set(values) for values in splits.values()))
    missing_ids = sorted(expected - all_split_ids)
    orphan_ids = sorted(all_split_ids - expected)
    valid = (
        not any(duplicate_ids.values())
        and not any(overlap.values())
        and not missing_ids
        and not orphan_ids
    )
    report = {
        "created_utc": utc_now(),
        "valid": valid,
        "counts": {name: len(values) for name, values in splits.items()},
        "duplicate_ids": duplicate_ids,
        "overlap": overlap,
        "missing_ids": missing_ids,
        "orphan_ids": orphan_ids,
    }
    write_json(output_path, report)
    if strict and not valid:
        raise ValueError(f"Split validation failed. See {output_path}")
    return report


def setup_experiment_logging(experiment_dir: Optional[Path], level: str = "INFO") -> logging.Logger:
    """Configure timestamped console and experiment-file logging."""
    logger = logging.getLogger("protliggnn")
    logger.handlers.clear()
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(numeric_level)
    logger.addHandler(stream_handler)
    if experiment_dir is not None:
        experiment_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(experiment_dir / "experiment.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    return logger


def write_environment_locks(experiment_dir: Path, environment: Optional[Mapping[str, object]] = None) -> Dict[str, str]:
    """Write environment.lock, requirements.lock, and pip_freeze.txt."""
    paths = {
        "environment_lock": experiment_dir / "environment.lock",
        "requirements_lock": experiment_dir / "requirements.lock",
        "pip_freeze": experiment_dir / "pip_freeze.txt",
    }
    write_json(paths["environment_lock"], dict(environment or capture_environment()))

    installed = []
    for dist in importlib_metadata.distributions():
        name = dist.metadata.get("Name") or dist.metadata.get("Summary") or "unknown"
        version = dist.version
        installed.append(f"{name}=={version}")
    paths["requirements_lock"].write_text("\n".join(sorted(installed, key=str.lower)) + "\n", encoding="utf-8")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        paths["pip_freeze"].write_text(result.stdout, encoding="utf-8")
    except Exception as exc:
        paths["pip_freeze"].write_text(f"pip freeze unavailable: {exc}\n", encoding="utf-8")

    return {name: str(path) for name, path in paths.items()}


def write_experiment_metadata(args, experiment_dir: Path) -> Path:
    tags = []
    if getattr(args, "tags", None):
        tags = [tag.strip() for tag in str(args.tags).split(",") if tag.strip()]
    payload = {
        "experiment_purpose": getattr(args, "experiment_purpose", "benchmark"),
        "research_question": getattr(args, "research_question", "Can ProtLigGNN reproducibly predict protein-ligand affinity under this benchmark configuration?"),
        "author": getattr(args, "author", "unknown"),
        "description": getattr(args, "description", ""),
        "notes": getattr(args, "notes", ""),
        "tags": tags,
        "expected_outcome": getattr(args, "expected_outcome", ""),
        "current_milestone": getattr(args, "current_milestone", "research-infrastructure"),
        "architecture_version": getattr(args, "architecture_version", "protliggnn-v1"),
        "research_version": getattr(args, "research_version", "benchmark-v2"),
        "created_utc": utc_now(),
    }
    path = experiment_dir / "metadata.json"
    write_json(path, payload)
    return path


def read_registry(registry_path: Path) -> List[Dict[str, str]]:
    if not registry_path.exists():
        return []
    with registry_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def update_experiment_registry(experiments_root: Path, row: Mapping[str, object]) -> Path:
    """Append or update one run row in experiments/registry.csv."""
    experiments_root.mkdir(parents=True, exist_ok=True)
    registry_path = experiments_root / "registry.csv"
    rows = read_registry(registry_path)
    row_id = str(row.get("Run ID", ""))
    normalized = {field: str(row.get(field, "")) for field in REGISTRY_FIELDS}
    replaced = False
    for idx, existing in enumerate(rows):
        if existing.get("Run ID") == row_id:
            rows[idx] = normalized
            replaced = True
            break
    if not replaced:
        rows.append(normalized)
    with registry_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return registry_path


def find_duplicate_experiments(
    experiments_root: Path,
    config_hash: str,
    dataset_hash: str,
    exclude_run_id: Optional[str] = None,
) -> List[Dict[str, str]]:
    duplicates = []
    for row in read_registry(experiments_root / "registry.csv"):
        if exclude_run_id and row.get("Run ID") == exclude_run_id:
            continue
        if row.get("Config Hash") == config_hash and row.get("Dataset Hash") == dataset_hash:
            if row.get("Experiment Status") in {"SUCCESS", "RUNNING"}:
                duplicates.append(row)
    return duplicates


def latest_run_dir(experiments_root: Path) -> Optional[Path]:
    candidates = [
        child
        for child in experiments_root.iterdir()
        if child.is_dir() and child.name.startswith("run_") and (child / "checkpoint.pt").exists()
    ] if experiments_root.exists() else []
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: path.stat().st_mtime)[-1]


def resolve_resume_reference(resume: str, experiments_root: Path) -> Dict[str, object]:
    """Resolve --resume latest, run_XXX, or a checkpoint path."""
    if resume == "latest":
        run_dir = latest_run_dir(experiments_root)
        if run_dir is None:
            raise FileNotFoundError(f"No resumable runs found under {experiments_root}")
        return {"checkpoint_path": run_dir / "checkpoint.pt", "run_dir": run_dir}
    candidate = Path(resume)
    if candidate.exists() and candidate.is_file():
        return {"checkpoint_path": candidate, "run_dir": candidate.parent}
    run_dir = experiments_root / resume
    checkpoint_path = run_dir / "checkpoint.pt"
    if checkpoint_path.exists():
        return {"checkpoint_path": checkpoint_path, "run_dir": run_dir}
    raise FileNotFoundError(f"Unable to resolve resume target: {resume}")


def validate_experiment_artifacts(
    experiment_dir: Path,
    required_files: Mapping[str, Path],
    required_dirs: Optional[Mapping[str, Path]] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, object]:
    required_dirs = required_dirs or {}
    file_records = {}
    missing = []
    for name, path in required_files.items():
        exists = path.exists() and path.is_file() and path.stat().st_size >= 0
        file_records[name] = {"path": str(path), "exists": exists, "size_bytes": path.stat().st_size if path.exists() else None}
        if not exists:
            missing.append(name)
    dir_records = {}
    for name, path in required_dirs.items():
        exists = path.exists() and path.is_dir()
        file_count = sum(1 for child in path.iterdir() if child.is_file()) if exists else 0
        valid = exists and file_count > 0
        dir_records[name] = {"path": str(path), "exists": exists, "file_count": file_count, "valid": valid}
        if not valid:
            missing.append(name)
    report = {
        "created_utc": utc_now(),
        "valid": not missing,
        "missing_or_invalid": missing,
        "files": file_records,
        "directories": dir_records,
    }
    path = output_path or (experiment_dir / "artifact_validation.json")
    write_json(path, report)
    return report


def write_reproducibility_report(
    experiment_dir: Path,
    artifact_validation: Mapping[str, object],
    output_path: Optional[Path] = None,
) -> Dict[str, object]:
    checks = {
        "Dataset Fingerprint": experiment_dir / "dataset_hash.json",
        "Configuration": experiment_dir / "config.json",
        "Config Hash": experiment_dir / "config_hash.txt",
        "Environment": experiment_dir / "environment.json",
        "Environment Freeze": experiment_dir / "requirements.lock",
        "Seeds": experiment_dir / "config.json",
        "Split": experiment_dir / "split" / "metadata.json",
        "Split Validation": experiment_dir / "split_validation.json",
        "Checkpoint": experiment_dir / "checkpoint.pt",
        "Manifest": experiment_dir / "manifest.json",
        "Metrics": experiment_dir / "metrics.json",
        "Predictions": experiment_dir / "predictions.csv",
        "Logs": experiment_dir / "experiment.log",
        "Artifacts": experiment_dir / "artifact_validation.json",
    }
    rows = []
    score = 0
    points_each = 100 / len(checks)
    recommendations = []
    for name, path in checks.items():
        ok = path.exists()
        if ok:
            score += points_each
        else:
            recommendations.append(f"Create or restore missing {name}: {path}")
        rows.append({"component": name, "ok": ok, "path": str(path)})
    if not artifact_validation.get("valid", False):
        recommendations.append("Resolve missing artifacts listed in artifact_validation.json.")
    score_int = int(round(score))
    report_path = output_path or (experiment_dir / "reproducibility_report.md")
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# Reproducibility Report\n\n")
        handle.write("| Component | Status | Artifact |\n")
        handle.write("| --- | --- | --- |\n")
        for row in rows:
            handle.write(f"| {row['component']} | {'PASS' if row['ok'] else 'FAIL'} | `{row['path']}` |\n")
        handle.write(f"\nOverall: {score_int} / 100\n\n")
        handle.write("## Recommendations\n\n")
        if recommendations:
            for item in recommendations:
                handle.write(f"- {item}\n")
        else:
            handle.write("- No missing reproducibility artifacts detected.\n")
    summary = {"score": score_int, "checks": rows, "recommendations": recommendations, "path": str(report_path)}
    write_json(experiment_dir / "reproducibility_report.json", summary)
    return summary


def write_experiment_lifecycle(experiment_dir: Path) -> Path:
    path = experiment_dir / "experiment_lifecycle.md"
    path.write_text(
        """# Experiment Lifecycle

```text
CLI
  |
  v
Configuration
  |
  v
Dataset
  |
  v
Fingerprint
  |
  v
Split
  |
  v
Validation
  |
  v
Training
  |
  v
Checkpoint
  |
  v
Evaluation
  |
  v
Metrics
  |
  v
Reports
  |
  v
Artifact Validation
  |
  v
Registry Update
  |
  v
Experiment Complete
```
""",
        encoding="utf-8",
    )
    return path


def registry_base_row(
    run_id: str,
    git_commit: object,
    dataset_version: str,
    dataset_hash: str,
    config_hash: str,
    split_strategy: str,
    seed: int,
    checkpoint_path: Path,
    status: str,
    duration_seconds: Optional[float] = None,
    best_validation_rmse: Optional[object] = None,
    test_metrics: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    return {
        "Run ID": run_id,
        "Timestamp": utc_now(),
        "Git Commit": git_commit or "",
        "Dataset Version": dataset_version,
        "Dataset Hash": dataset_hash,
        "Config Hash": config_hash,
        "Split Strategy": split_strategy,
        "Seed": seed,
        "Best Validation RMSE": "" if best_validation_rmse is None else best_validation_rmse,
        "Test Metrics": json.dumps(dict(test_metrics or {}), sort_keys=True, default=str),
        "Experiment Status": status,
        "Duration": "" if duration_seconds is None else f"{duration_seconds:.3f}",
        "Checkpoint Path": str(checkpoint_path),
    }
