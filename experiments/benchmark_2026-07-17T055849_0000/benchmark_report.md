# protliggnn_benchmark

Generated UTC: 2026-07-17T06:00:41+00:00

## Aggregate Metrics

| Metric | Mean +/- Std | CI95 | Bootstrap CI95 | Best | Worst | N |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mse | 0.644329 +/- 0.622769 | [0.098448, 1.190210] | [0.184659, 1.170754] | 0.082963 | 1.532892 | 5 |
| rmse | 0.718962 +/- 0.399097 | [0.369138, 1.068786] | [0.414024, 1.038157] | 0.288033 | 1.238100 | 5 |
| mae | 0.718962 +/- 0.399097 | [0.369138, 1.068786] | [0.414024, 1.038157] | 0.288033 | 1.238100 | 5 |
| median_absolute_error | 0.718962 +/- 0.399097 | [0.369138, 1.068786] | [0.414024, 1.038157] | 0.288033 | 1.238100 | 5 |
| bias | -0.423377 +/- 0.762464 | [-1.091706, 0.244952] | [-1.023900, 0.177146] | -1.238100 | 0.450930 | 5 |
| mean_prediction | 4.277847 +/- 0.991122 | [3.409091, 5.146604] | [3.505776, 5.021853] | 2.983749 | 5.453562 | 5 |

## Paired Statistical Tests

No paired baseline metrics were supplied to this benchmark runner, so paired tests are not applicable. The aggregation code preserves per-seed rows so future baseline-vs-candidate paired tests can be added without changing the experiment artifact format.

## Per-Seed Runs

| Seed | Run Dir | RMSE | MAE | Pearson | Spearman | R2 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 42 | experiments\run_037 | 1.238100 | 1.238100 | nan | nan | nan |
| 123 | experiments\run_038 | 1.025552 | 1.025552 | nan | nan | nan |
| 777 | experiments\run_039 | 0.592196 | 0.592196 | nan | nan | nan |
| 2024 | experiments\run_040 | 0.450930 | 0.450930 | nan | nan | nan |
| 3407 | experiments\run_041 | 0.288033 | 0.288033 | nan | nan | nan |
