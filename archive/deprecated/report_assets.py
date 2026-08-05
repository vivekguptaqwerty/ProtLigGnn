import csv
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd


OUTPUTS_DIR = Path("outputs")
LOGS_DIR = OUTPUTS_DIR / "logs"
REPORT_DIR = OUTPUTS_DIR / "report_assets"


def read_text_auto(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8", "utf-16", "utf-16-le", "utf-16-be", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def parse_log_metrics(log_path: Path) -> Optional[Dict[str, object]]:
    text = read_text_auto(log_path)
    lines = text.splitlines()
    best_epoch_line = next((line for line in reversed(lines) if "Loaded best model from epoch" in line), "")
    test_line = next((line for line in reversed(lines) if "Test loss=" in line), "")

    best_epoch_match = re.search(r"Loaded best model from epoch\s+(\d+)", best_epoch_line)
    test_match = re.search(
        r"Test loss=([\-0-9.]+)\s+PCC=([\-0-9.]+)\s+Spearman=([\-0-9.]+)\s+RMSE=([\-0-9.]+)\s+MAE=([\-0-9.]+)",
        test_line,
    )

    if not test_line or not test_match:
        return None

    run_name = log_path.stem
    no_crossgraph = "no_crossgraph" in run_name
    default_command = [
        "python protliggnn_train.py",
        "--data_dir data/pdbbind2020",
        "--device cpu",
    ]
    if no_crossgraph:
        default_command.append("--no_crossgraph")
    default_command.append(f"--run_name {run_name}")

    return {
        "run_name": run_name,
        "log_file": str(log_path),
        "command": " ".join(default_command),
        "best_epoch": int(best_epoch_match.group(1)) if best_epoch_match else None,
        "test_loss": float(test_match.group(1)),
        "test_pcc": float(test_match.group(2)),
        "test_spearman": float(test_match.group(3)),
        "test_rmse": float(test_match.group(4)),
        "test_mae": float(test_match.group(5)),
    }


def collect_results() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if LOGS_DIR.exists():
        for log_path in sorted(LOGS_DIR.glob("*.log")):
            parsed = parse_log_metrics(log_path)
            if parsed is not None:
                rows.append(parsed)
    return pd.DataFrame(rows)


def save_results_table(results_df: pd.DataFrame) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / "results_table.csv"
    if results_df.empty:
        pd.DataFrame(
            columns=[
                "run_name",
                "command",
                "best_epoch",
                "test_loss",
                "test_pcc",
                "test_spearman",
                "test_rmse",
                "test_mae",
                "log_file",
            ]
        ).to_csv(output_path, index=False)
    else:
        results_df.to_csv(output_path, index=False)
    return output_path


def save_ablation_plot(results_df: pd.DataFrame) -> Optional[Path]:
    if results_df.empty:
        return None

    plot_df = results_df.copy()
    metrics = ["test_pcc", "test_spearman", "test_rmse", "test_mae"]
    labels = ["PCC", "Spearman", "RMSE", "MAE"]
    x = range(len(metrics))
    width = 0.8 / max(len(plot_df), 1)

    plt.figure(figsize=(10, 5))
    for idx, row in enumerate(plot_df.itertuples(index=False)):
        offsets = [value + (idx - (len(plot_df) - 1) / 2) * width for value in x]
        plt.bar(offsets, [getattr(row, metric) for metric in metrics], width=width, label=row.run_name)

    plt.xticks(list(x), labels)
    plt.ylabel("Metric value")
    plt.title("Ablation Comparison")
    plt.legend()
    plt.tight_layout()

    output_path = REPORT_DIR / "ablation_comparison.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_training_loss_curve() -> Optional[Path]:
    history_files = sorted(OUTPUTS_DIR.glob("epoch_history_*.csv"))
    if not history_files:
        return None

    plt.figure(figsize=(9, 5))
    for history_file in history_files:
        run_name = history_file.stem.replace("epoch_history_", "")
        df = pd.read_csv(history_file)
        if {"epoch", "train_loss", "val_loss"}.issubset(df.columns):
            plt.plot(df["epoch"], df["train_loss"], label=f"{run_name} train")
            plt.plot(df["epoch"], df["val_loss"], linestyle="--", label=f"{run_name} val")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.tight_layout()

    output_path = REPORT_DIR / "training_loss_curve.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_predicted_vs_true_scatter() -> Optional[Path]:
    prediction_files = sorted(OUTPUTS_DIR.glob("test_predictions_*.csv"))
    if not prediction_files:
        return None

    plt.figure(figsize=(7, 7))
    min_val = None
    max_val = None
    for prediction_file in prediction_files:
        run_name = prediction_file.stem.replace("test_predictions_", "")
        df = pd.read_csv(prediction_file)
        if {"true_affinity", "predicted_affinity"}.issubset(df.columns):
            plt.scatter(
                df["true_affinity"],
                df["predicted_affinity"],
                alpha=0.7,
                label=run_name,
            )
            local_min = min(df["true_affinity"].min(), df["predicted_affinity"].min())
            local_max = max(df["true_affinity"].max(), df["predicted_affinity"].max())
            min_val = local_min if min_val is None else min(min_val, local_min)
            max_val = local_max if max_val is None else max(max_val, local_max)

    if min_val is None or max_val is None:
        plt.close()
        return None

    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=1.0, color="tab:red")
    plt.xlabel("True affinity")
    plt.ylabel("Predicted affinity")
    plt.title("Predicted vs True Affinity")
    plt.legend()
    plt.tight_layout()

    output_path = REPORT_DIR / "predicted_vs_true_scatter.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_short_summary(results_df: pd.DataFrame, generated_files: List[Path]) -> Path:
    lines: List[str] = []
    lines.append("ProtLigGNN Report Assets Summary")
    lines.append("==============================")
    lines.append("")

    if results_df.empty:
        lines.append("No experiment logs with parseable test metrics were found.")
    else:
        lines.append(f"Parsed {len(results_df)} experiment log(s).")
        best_pcc_row = results_df.loc[results_df["test_pcc"].idxmax()]
        best_rmse_row = results_df.loc[results_df["test_rmse"].idxmin()]
        lines.append(
            f"Best PCC run: {best_pcc_row['run_name']} "
            f"(PCC={best_pcc_row['test_pcc']:.4f}, Spearman={best_pcc_row['test_spearman']:.4f})"
        )
        lines.append(
            f"Best RMSE run: {best_rmse_row['run_name']} "
            f"(RMSE={best_rmse_row['test_rmse']:.4f}, MAE={best_rmse_row['test_mae']:.4f})"
        )
        cross_rows = results_df[results_df["run_name"].str.contains("crossgraph") & ~results_df["run_name"].str.contains("no_crossgraph")]
        no_cross_rows = results_df[results_df["run_name"].str.contains("no_crossgraph")]
        if not cross_rows.empty and not no_cross_rows.empty:
            lines.append("Crossgraph and no-crossgraph runs are both present for ablation comparison.")

    lines.append("")
    lines.append("Generated files:")
    for path in generated_files:
        lines.append(f"- {path}")

    output_path = REPORT_DIR / "short_results_summary.txt"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results_df = collect_results()
    generated_files: List[Path] = []

    generated_files.append(save_results_table(results_df))
    ablation_path = save_ablation_plot(results_df)
    if ablation_path is not None:
        generated_files.append(ablation_path)
    history_path = save_training_loss_curve()
    if history_path is not None:
        generated_files.append(history_path)
    scatter_path = save_predicted_vs_true_scatter()
    if scatter_path is not None:
        generated_files.append(scatter_path)
    generated_files.append(save_short_summary(results_df, generated_files.copy()))

    print(f"Saved report assets to {REPORT_DIR.resolve()}")
    for path in generated_files:
        print(path.name)


if __name__ == "__main__":
    main()
