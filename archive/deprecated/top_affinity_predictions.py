import csv
from pathlib import Path
from typing import Dict, List


OUTPUT_DIRS = [Path("outputs"), Path("outputs/runs")]
REPORT_DIR = Path("outputs/report_assets")
REQUIRED_COLUMNS = {"pdb_id", "true_affinity", "predicted_affinity", "error"}


def find_prediction_files() -> List[Path]:
    files: List[Path] = []
    for base_dir in OUTPUT_DIRS:
        if base_dir.exists():
            files.extend(sorted(base_dir.rglob("test_predictions_*.csv")))
    return files


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(set(reader.fieldnames)):
            raise ValueError(f"Missing required columns in {path}")
        return list(reader)


def collect_ranked_predictions() -> List[Dict[str, object]]:
    ranked_rows: List[Dict[str, object]] = []
    for path in find_prediction_files():
        run_name = path.stem.replace("test_predictions_", "")
        try:
            rows = load_rows(path)
        except Exception as exc:
            print(f"Warning: skipping {path} ({exc})")
            continue

        for row in rows:
            predicted_affinity = float(row["predicted_affinity"])
            true_affinity = float(row["true_affinity"])
            absolute_error = abs(float(row["error"]))
            ranked_rows.append(
                {
                    "run_name": run_name,
                    "pdb_id": row["pdb_id"],
                    # PDBbind affinity labels are larger for stronger binders in this setup,
                    # so we rank descending by predicted_affinity to surface the strongest predictions.
                    "predicted_affinity": predicted_affinity,
                    "true_affinity": true_affinity,
                    "absolute_error": absolute_error,
                }
            )
    ranked_rows.sort(key=lambda row: row["predicted_affinity"], reverse=True)
    return ranked_rows


def save_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "run_name",
                "pdb_id",
                "predicted_affinity",
                "true_affinity",
                "absolute_error",
            ]
        )
        for idx, row in enumerate(rows, start=1):
            writer.writerow(
                [
                    idx,
                    row["run_name"],
                    row["pdb_id"],
                    f"{row['predicted_affinity']:.6f}",
                    f"{row['true_affinity']:.6f}",
                    f"{row['absolute_error']:.6f}",
                ]
            )


def save_text(rows: List[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Top 10 Predicted Binding Affinities",
        "===================================",
        "",
        f"{'Rank':<6}{'Run Name':<34}{'PDB ID':<10}{'Predicted':<14}{'True':<14}{'Abs Error':<14}",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.append(
            f"{idx:<6}{row['run_name']:<34}{row['pdb_id']:<10}"
            f"{row['predicted_affinity']:<14.4f}{row['true_affinity']:<14.4f}{row['absolute_error']:<14.4f}"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_top10(rows: List[Dict[str, object]]) -> None:
    print("Top 10 strongest predicted affinities")
    print("Rank  Run Name                          PDB ID    Predicted      True           Abs Error")
    for idx, row in enumerate(rows, start=1):
        print(
            f"{idx:<5} {row['run_name']:<32} {row['pdb_id']:<8} "
            f"{row['predicted_affinity']:<14.4f} {row['true_affinity']:<14.4f} {row['absolute_error']:<14.4f}"
        )


def main() -> None:
    ranked_rows = collect_ranked_predictions()
    if not ranked_rows:
        print("No valid test_predictions_*.csv files found.")
        return

    top10 = ranked_rows[:10]
    save_csv(top10, REPORT_DIR / "top10_predicted_affinities.csv")
    save_text(top10, REPORT_DIR / "top10_predicted_affinities.txt")
    print_top10(top10)
    print(f"Saved outputs to {REPORT_DIR.resolve()}")


if __name__ == "__main__":
    main()
