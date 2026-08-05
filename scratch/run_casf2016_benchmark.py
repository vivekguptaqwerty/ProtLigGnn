from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
CORESET_PATH = ROOT / "data" / "CASF-2016" / "power_scoring" / "CoreSet.dat"
PREDICTIONS_PATH = ROOT / "reports" / "CASF_Evaluation" / "raw_predictions" / "raw_predictions.csv"


def read_coreset(path: Path) -> pd.DataFrame:
    rows: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") and not line.startswith("#code"):
            continue
        rows.append(line)
    tmp = "\n".join(rows)
    from io import StringIO

    df = pd.read_csv(StringIO(tmp), sep=r"\s+", engine="python")
    return df.drop_duplicates(subset=["#code"], keep="first")


def predictive_index(group: pd.DataFrame) -> float:
    sorted_group = group.sort_values("logKa", ascending=True)
    weights: list[float] = []
    concordant_weights: list[float] = []
    rows = list(sorted_group.itertuples(index=False))
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            low = rows[i]
            high = rows[j]
            weight = abs(float(low.logKa) - float(high.logKa))
            weights.append(weight)
            if float(low.score) < float(high.score):
                concordant_weights.append(weight)
            elif float(low.score) > float(high.score):
                concordant_weights.append(-weight)
            else:
                concordant_weights.append(0.0)
    return float(sum(concordant_weights) / sum(weights))


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric,
    *,
    n_bootstrap: int = 1000,
    seed: int = 3407,
) -> list[float]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        values.append(float(metric(y_true[idx], y_pred[idx])))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CASF-2016 scoring and ranking power from prediction CSV.")
    parser.add_argument("--coreset", default=str(CORESET_PATH))
    parser.add_argument("--predictions", default=str(PREDICTIONS_PATH))
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else ROOT / "scratch" / f"casf2016_benchmark_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    coreset_path = Path(args.coreset)
    predictions_path = Path(args.predictions)
    coreset = read_coreset(coreset_path)
    preds = pd.read_csv(predictions_path)
    scores = preds[["pdb_id", "predicted_affinity"]].rename(
        columns={"pdb_id": "#code", "predicted_affinity": "score"}
    )
    scores.to_csv(out_dir / "ligprotgnn_casf_scores.dat", sep="\t", index=False)

    merged = coreset.merge(scores, on="#code", how="inner")
    merged = merged[merged["score"] > 0].copy()
    merged.to_csv(out_dir / "scoring_power_processed_score.tsv", sep="\t", index=False)

    x = merged[["score"]].to_numpy()
    y = merged["logKa"].to_numpy()
    model = LinearRegression().fit(x, y)
    fitted = model.predict(x)

    rmse = float(mean_squared_error(y, merged["score"].to_numpy()) ** 0.5)
    mae = float(mean_absolute_error(y, merged["score"].to_numpy()))
    pearson = float(stats.pearsonr(y, merged["score"].to_numpy()).statistic)
    spearman = float(stats.spearmanr(y, merged["score"].to_numpy()).statistic)
    kendall = float(stats.kendalltau(y, merged["score"].to_numpy()).statistic)
    fit_sd = float((mean_squared_error(y, fitted) * len(merged) / (len(merged) - 1)) ** 0.5)
    r2 = float(r2_score(y, merged["score"].to_numpy()))

    ranking_rows = []
    success_at_1 = 0
    success_at_2 = 0
    success_at_3 = 0

    for target, group in merged.groupby("target"):
        if len(group) != 5:
            continue
        true_best = str(group.sort_values("logKa", ascending=False).iloc[0]["#code"])
        ranked = group.sort_values("score", ascending=False).reset_index(drop=True)
        ranked_codes = [str(code) for code in ranked["#code"].tolist()]
        if true_best in ranked_codes[:1]:
            success_at_1 += 1
        if true_best in ranked_codes[:2]:
            success_at_2 += 1
        if true_best in ranked_codes[:3]:
            success_at_3 += 1
        ranking_rows.append(
            {
                "target": int(target),
                "representative_code": true_best,
                "rank1": ranked_codes[0],
                "rank2": ranked_codes[1],
                "rank3": ranked_codes[2],
                "rank4": ranked_codes[3],
                "rank5": ranked_codes[4],
                "spearman": float(stats.spearmanr(ranked["logKa"], ranked["score"]).statistic),
                "kendall": float(stats.kendalltau(ranked["logKa"], ranked["score"]).statistic),
                "predictive_index": predictive_index(ranked),
            }
        )

    ranking = pd.DataFrame(ranking_rows)
    ranking.to_csv(out_dir / "ranking_power_results.tsv", sep="\t", index=False)

    n_groups = len(ranking)
    metrics = {
        "run_timestamp": timestamp,
        "inputs": {
            "coreset": str(coreset_path),
            "predictions": str(predictions_path),
            "preference": "positive",
        },
        "coverage": {
            "official_coreset_rows": int(len(coreset)),
            "prediction_rows": int(len(preds)),
            "merged_rows": int(len(merged)),
            "ranking_groups": int(n_groups),
        },
        "scoring_power": {
            "pearson_r": pearson,
            "pearson_r_95ci": bootstrap_ci(
                y,
                merged["score"].to_numpy(),
                lambda a, b: stats.pearsonr(a, b).statistic,
                n_bootstrap=args.bootstrap,
            ),
            "standard_deviation_of_fit": fit_sd,
            "regression_intercept": float(model.intercept_),
            "regression_slope": float(model.coef_[0]),
            "spearman": spearman,
            "kendall_tau": kendall,
            "direct_rmse": rmse,
            "direct_rmse_95ci": bootstrap_ci(
                y,
                merged["score"].to_numpy(),
                lambda a, b: mean_squared_error(a, b) ** 0.5,
                n_bootstrap=args.bootstrap,
            ),
            "mae": mae,
            "r2_direct": r2,
        },
        "ranking_power": {
            "mean_spearman": float(ranking["spearman"].mean()),
            "mean_kendall_tau": float(ranking["kendall"].mean()),
            "mean_predictive_index": float(ranking["predictive_index"].mean()),
            "top1_success_rate": float(success_at_1 / n_groups),
            "top2_success_rate": float(success_at_2 / n_groups),
            "top3_success_rate": float(success_at_3 / n_groups),
        },
        "not_run": {
            "docking_power": "requires pose-level score files for CASF decoys_docking",
            "forward_screening_power": "requires target-ligand decoy score files for CASF decoys_screening",
            "reverse_screening_power": "requires ligand-target decoy score files for CASF decoys_screening",
        },
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    summary_lines = [
        "CASF-2016 benchmark completed",
        f"Output directory: {out_dir}",
        "",
        f"Coverage: {len(merged)}/{len(coreset)} coreset complexes, {n_groups} ranking clusters",
        "",
        "Scoring power:",
        f"  Pearson R: {pearson:.3f}",
        f"  Standard deviation of fit: {fit_sd:.2f}",
        f"  Spearman: {spearman:.3f}",
        f"  Kendall tau: {kendall:.3f}",
        f"  Direct RMSE: {rmse:.3f}",
        f"  MAE: {mae:.3f}",
        f"  Direct R^2: {r2:.3f}",
        "",
        "Ranking power:",
        f"  Mean Spearman: {metrics['ranking_power']['mean_spearman']:.3f}",
        f"  Mean Kendall tau: {metrics['ranking_power']['mean_kendall_tau']:.3f}",
        f"  Mean Predictive Index: {metrics['ranking_power']['mean_predictive_index']:.3f}",
        f"  Top-1 success: {metrics['ranking_power']['top1_success_rate'] * 100:.2f}%",
        f"  Top-2 success: {metrics['ranking_power']['top2_success_rate'] * 100:.2f}%",
        f"  Top-3 success: {metrics['ranking_power']['top3_success_rate'] * 100:.2f}%",
        "",
        "Docking/screening power not run: decoy-level model score files were not present.",
    ]
    summary = "\n".join(summary_lines)
    (out_dir / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
