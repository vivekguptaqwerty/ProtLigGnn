"""Compare saved ProtLigGNN experiment runs from artifact directories."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from benchmarking import METRIC_KEYS, read_json, utc_now
from experiment_infra import canonical_config, read_registry


def parse_args():
    parser = argparse.ArgumentParser(description="Compare ProtLigGNN experiment runs.")
    parser.add_argument("runs", nargs="+", help="Run IDs such as run_001 or paths to run directories.")
    parser.add_argument("--experiments_root", type=str, default="experiments")
    parser.add_argument("--out_dir", type=str, default=None)
    return parser.parse_args()


def resolve_run(ref: str, experiments_root: Path) -> Path:
    path = Path(ref)
    if path.exists():
        return path
    candidate = experiments_root / ref
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Run not found: {ref}")


def safe_read_json(path: Path) -> Dict:
    return read_json(path) if path.exists() else {}


def read_training_summary(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {"epoch_count": 0, "best_val_rmse": None, "last_val_rmse": None}
    rows = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    val_rmse = []
    for row in rows:
        try:
            val_rmse.append(float(row.get("val_rmse", "")))
        except ValueError:
            continue
    return {
        "epoch_count": len(rows),
        "best_val_rmse": min(val_rmse) if val_rmse else None,
        "last_val_rmse": val_rmse[-1] if val_rmse else None,
    }


def registry_by_run_id(experiments_root: Path) -> Dict[str, Dict[str, str]]:
    return {row.get("Run ID", ""): row for row in read_registry(experiments_root / "registry.csv")}


def summarize_run(run_dir: Path, registry_rows: Mapping[str, Mapping[str, str]]) -> Dict[str, object]:
    metrics = safe_read_json(run_dir / "metrics.json")
    config = safe_read_json(run_dir / "config.json")
    split = safe_read_json(run_dir / "split" / "metadata.json")
    environment = safe_read_json(run_dir / "environment.json")
    metadata = safe_read_json(run_dir / "metadata.json")
    dataset_hash = safe_read_json(run_dir / "dataset_hash.json")
    config_hash = (run_dir / "config_hash.txt").read_text(encoding="utf-8").strip() if (run_dir / "config_hash.txt").exists() else ""
    training = read_training_summary(run_dir / "training_history.csv")
    registry = dict(registry_rows.get(run_dir.name, {}))
    return {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "metrics": metrics,
        "config": config,
        "canonical_config": canonical_config(config),
        "split": split,
        "environment": environment,
        "metadata": metadata,
        "dataset_hash": dataset_hash,
        "config_hash": config_hash,
        "training": training,
        "registry": registry,
    }


def value_at(summary: Mapping[str, object], path: Sequence[str]):
    value = summary
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def write_metric_csv(summaries: Sequence[Mapping[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["category", "key", "run_id", "value"])
        for summary in summaries:
            run_id = summary["run_id"]
            for metric in METRIC_KEYS + ["loss", "best_epoch", "test_count"]:
                writer.writerow(["metric", metric, run_id, value_at(summary, ["metrics", metric])])
            for key in sorted(summary["canonical_config"]):
                writer.writerow(["config", key, run_id, summary["canonical_config"][key]])
            writer.writerow(["split", "effective_strategy", run_id, value_at(summary, ["split", "effective_strategy"])])
            writer.writerow(["split", "train_count", run_id, value_at(summary, ["split", "counts", "train"])])
            writer.writerow(["split", "val_count", run_id, value_at(summary, ["split", "counts", "val"])])
            writer.writerow(["split", "test_count", run_id, value_at(summary, ["split", "counts", "test"])])
            writer.writerow(["runtime", "duration", run_id, value_at(summary, ["registry", "Duration"])])
            writer.writerow(["fingerprint", "config_hash", run_id, summary["config_hash"]])
            writer.writerow(["fingerprint", "dataset_hash", run_id, value_at(summary, ["dataset_hash", "overall_sha256"])])


def collect_config_differences(summaries: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    keys = sorted({key for summary in summaries for key in summary["canonical_config"].keys()})
    differences = []
    for key in keys:
        values = [summary["canonical_config"].get(key) for summary in summaries]
        if len({json.dumps(value, sort_keys=True, default=str) for value in values}) > 1:
            differences.append({"key": key, "values": values})
    return differences


def write_markdown_report(summaries: Sequence[Mapping[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config_diffs = collect_config_differences(summaries)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Experiment Comparison\n\n")
        handle.write(f"Generated UTC: {utc_now()}\n\n")
        handle.write("## Metrics\n\n")
        handle.write("| Metric | " + " | ".join(summary["run_id"] for summary in summaries) + " |\n")
        handle.write("| --- | " + " | ".join("---:" for _ in summaries) + " |\n")
        for metric in METRIC_KEYS + ["loss", "best_epoch"]:
            values = [value_at(summary, ["metrics", metric]) for summary in summaries]
            handle.write("| " + metric + " | " + " | ".join(str(value) for value in values) + " |\n")

        handle.write("\n## Hyperparameter Differences\n\n")
        if config_diffs:
            handle.write("| Key | " + " | ".join(summary["run_id"] for summary in summaries) + " |\n")
            handle.write("| --- | " + " | ".join("---" for _ in summaries) + " |\n")
            for diff in config_diffs:
                handle.write("| " + diff["key"] + " | " + " | ".join(str(value) for value in diff["values"]) + " |\n")
        else:
            handle.write("No canonical configuration differences detected.\n")

        handle.write("\n## Splits\n\n")
        handle.write("| Run | Strategy | Train | Val | Test | Dataset Hash |\n")
        handle.write("| --- | --- | ---: | ---: | ---: | --- |\n")
        for summary in summaries:
            handle.write(
                f"| {summary['run_id']} | {value_at(summary, ['split', 'effective_strategy'])} | "
                f"{value_at(summary, ['split', 'counts', 'train'])} | {value_at(summary, ['split', 'counts', 'val'])} | "
                f"{value_at(summary, ['split', 'counts', 'test'])} | {value_at(summary, ['dataset_hash', 'overall_sha256'])} |\n"
            )

        handle.write("\n## Training Curves\n\n")
        handle.write("| Run | Epochs | Best Val RMSE | Last Val RMSE |\n")
        handle.write("| --- | ---: | ---: | ---: |\n")
        for summary in summaries:
            training = summary["training"]
            handle.write(
                f"| {summary['run_id']} | {training['epoch_count']} | {training['best_val_rmse']} | {training['last_val_rmse']} |\n"
            )

        handle.write("\n## Runtime And Environment\n\n")
        handle.write("| Run | Duration | Python | Torch | CUDA | GPU |\n")
        handle.write("| --- | ---: | --- | --- | --- | --- |\n")
        for summary in summaries:
            environment = summary["environment"]
            gpu = value_at(summary, ["environment", "hardware", "gpu"])
            handle.write(
                f"| {summary['run_id']} | {value_at(summary, ['registry', 'Duration'])} | "
                f"{value_at(summary, ['environment', 'python', 'version'])} | "
                f"{value_at(summary, ['environment', 'torch_runtime', 'version'])} | "
                f"{value_at(summary, ['environment', 'torch_runtime', 'cuda_version'])} | {gpu} |\n"
            )

        handle.write("\n## Statistical Interpretation\n\n")
        handle.write(
            "This utility compares saved experiment artifacts. For individual run directories, metric deltas are descriptive only; "
            "they are not statistically meaningful without matched multi-seed distributions or paired baseline rows. "
            "Use `benchmark.py --seeds 5` and compare aggregate benchmark reports for statistical claims.\n"
        )


def print_console_table(summaries: Sequence[Mapping[str, object]]) -> None:
    headers = ["run", "rmse", "mae", "pcc", "spearman", "r2", "seed"]
    print(" | ".join(headers))
    print(" | ".join("---" for _ in headers))
    for summary in summaries:
        metrics = summary["metrics"]
        row = [
            str(summary["run_id"]),
            str(metrics.get("rmse")),
            str(metrics.get("mae")),
            str(metrics.get("pcc")),
            str(metrics.get("spearman")),
            str(metrics.get("r2")),
            str(metrics.get("seed")),
        ]
        print(" | ".join(row))


def main() -> int:
    args = parse_args()
    experiments_root = Path(args.experiments_root)
    run_dirs = [resolve_run(ref, experiments_root) for ref in args.runs]
    registry_rows = registry_by_run_id(experiments_root)
    summaries = [summarize_run(run_dir, registry_rows) for run_dir in run_dirs]
    output_dir = Path(args.out_dir) if args.out_dir else experiments_root / "comparisons"
    safe_name = "_vs_".join(summary["run_id"] for summary in summaries)
    csv_path = output_dir / f"{safe_name}.csv"
    md_path = output_dir / f"{safe_name}.md"
    write_metric_csv(summaries, csv_path)
    write_markdown_report(summaries, md_path)
    print_console_table(summaries)
    print(f"CSV: {csv_path}")
    print(f"Report: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
