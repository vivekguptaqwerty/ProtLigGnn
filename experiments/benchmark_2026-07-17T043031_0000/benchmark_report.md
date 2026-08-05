# protliggnn_benchmark

Generated UTC: 2026-07-17T04:33:51+00:00

## Aggregate Metrics

| Metric | Mean +/- Std | CI95 | Bootstrap CI95 | Best | Worst | N |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mse | 1.997036 +/- 1.596754 | [0.597419, 3.396653] | [0.866523, 3.398724] | 0.266856 | 4.372682 | 5 |
| rmse | 1.308593 +/- 0.596468 | [0.785766, 1.831421] | [0.864633, 1.776193] | 0.516582 | 2.091096 | 5 |
| mae | 1.308593 +/- 0.596468 | [0.785766, 1.831421] | [0.864633, 1.776193] | 0.516582 | 2.091096 | 5 |
| median_absolute_error | 1.308593 +/- 0.596468 | [0.785766, 1.831421] | [0.864633, 1.776193] | 0.516582 | 2.091096 | 5 |
| bias | -1.101961 +/- 0.989140 | [-1.968980, -0.234941] | [-1.744729, -0.257776] | -2.091096 | 0.516582 | 5 |
| mean_prediction | 3.599263 +/- 1.263332 | [2.491904, 4.706622] | [2.644864, 4.528295] | 1.832264 | 4.865272 | 5 |

## Paired Statistical Tests

No paired baseline metrics were supplied to this benchmark runner, so paired tests are not applicable. The aggregation code preserves per-seed rows so future baseline-vs-candidate paired tests can be added without changing the experiment artifact format.

## Per-Seed Runs

| Seed | Run Dir | RMSE | MAE | Pearson | Spearman | R2 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 42 | experiments\run_017 | 0.516582 | 0.516582 | nan | nan | nan |
| 123 | experiments\run_018 | 1.658137 | 1.658137 | nan | nan | nan |
| 777 | experiments\run_019 | 1.180485 | 1.180485 | nan | nan | nan |
| 2024 | experiments\run_020 | 2.091096 | 2.091096 | nan | nan | nan |
| 3407 | experiments\run_021 | 1.096667 | 1.096667 | nan | nan | nan |
