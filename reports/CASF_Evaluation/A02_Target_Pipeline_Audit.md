# A02 Target Pipeline Audit: CASF-2016

## Transformation Audit
1. **Raw Affinities**: Verified that raw affinities are read from `CoreSet.dat` and correctly parsed without any modification.
2. **Log Transformations**: Verified that the affinities in `CoreSet.dat` are already log-transformed experimental values ($-\log_{10}(K_d)$ or $-\log_{10}(K_i)$).
3. **Scaling Normalization**: Confirmed that `protliggnn_train.py` does **not** employ any `StandardScaler`, `MinMaxScaler`, or z-score normalization on the target labels.
4. **Inverse Transformations**: Since no normalization or scaling is applied during training, no inverse transform is required during evaluation.
5. **Verdict**: **No target pipeline or scaling mismatch exists**. The evaluation pipeline processes the labels exactly as they were trained.
