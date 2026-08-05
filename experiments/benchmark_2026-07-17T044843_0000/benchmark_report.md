# protliggnn_benchmark

Generated UTC: 2026-07-17T04:54:23+00:00

## Aggregate Metrics

| Metric | Mean +/- Std | CI95 | Bootstrap CI95 | Best | Worst | N |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mse | 1.428655 +/- 0.880527 | [0.656839, 2.200471] | [0.778186, 2.131090] | 0.575511 | 2.584941 | 5 |
| rmse | 1.148939 +/- 0.368431 | [0.825995, 1.471883] | [0.871290, 1.434521] | 0.758625 | 1.607775 | 5 |
| mae | 1.148939 +/- 0.368431 | [0.825995, 1.471883] | [0.871290, 1.434521] | 0.758625 | 1.607775 | 5 |
| median_absolute_error | 1.148939 +/- 0.368431 | [0.825995, 1.471883] | [0.871290, 1.434521] | 0.758625 | 1.607775 | 5 |
| bias | -0.845490 +/- 0.944591 | [-1.673460, -0.017519] | [-1.434521, -0.042306] | -1.607775 | 0.758625 | 5 |
| mean_prediction | 3.855735 +/- 1.161474 | [2.837658, 4.873811] | [2.961502, 4.744968] | 2.315584 | 5.005470 | 5 |

## Paired Statistical Tests

No paired baseline metrics were supplied to this benchmark runner, so paired tests are not applicable. The aggregation code preserves per-seed rows so future baseline-vs-candidate paired tests can be added without changing the experiment artifact format.

## Per-Seed Runs

| Seed | Run Dir | RMSE | MAE | Pearson | Spearman | R2 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 42 | experiments\run_027 | 0.758625 | 0.758625 | nan | nan | nan |
| 123 | experiments\run_028 | 1.458383 | 1.458383 | nan | nan | nan |
| 777 | experiments\run_029 | 1.040287 | 1.040287 | nan | nan | nan |
| 2024 | experiments\run_030 | 1.607775 | 1.607775 | nan | nan | nan |
| 3407 | experiments\run_031 | 0.879627 | 0.879627 | nan | nan | nan |
