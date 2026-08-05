# Document 2: Complete Experimental Results, Benchmarks & Statistical Analysis

This document provides a comprehensive record of all experimental results, seed-level outputs, biostatistical tests, and computational footprints collected across the development cycles of the **LigProtGNN Research Suite**.

---

## 1. Seed-by-Seed RMSE Database

All models were evaluated across 5 seeds on the PDBBind local test set:

| Version | Seed 42 | Seed 123 | Seed 777 | Seed 2024 | Seed 3407 | Mean RMSE | Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline v1.0** | 1.66258 | 1.56176 | 1.57357 | 1.57835 | 1.49775 | 1.57481 | 0.05924 |
| **Optimized v1.1**| 1.58627 | 1.62862 | 1.51388 | 1.66585 | 1.62913 | 1.60476 | 0.05808 |
| **Geometry v1.2** | 1.46160 | 1.57270 | 1.53890 | 1.66710 | 1.48100 | 1.54426 | 0.08180 |
| **Attention v1.3**| 1.53724 | 1.55638 | 1.45884 | 1.66259 | 1.54933 | 1.55288 | 0.07271 |
| **IRM v1.4 (Cand)**| 1.61771 | 1.51856 | 1.63919 | 1.62963 | 1.36237 | 1.55349 | 0.11727 |

---

## 2. Deep Biostatistical Comparison

All pairwise statistics represent comparisons of the Candidate **IRM v1.4** against previous baselines:

### 2.1 Candidate v1.4 vs. Current Active Baseline v1.3 (Phase 3.1B)
- **Mean Difference**: $+0.00062$
- **Relative Difference**: **-0.04%** (Slight regression in mean error)
- **Paired t-test**:
  - \(t\)-statistic: \(0.00995\)
  - \(p\)-value: \(0.992538\) (Completely non-significant)
- **Wilcoxon Signed-Rank Test**:
  - \(w\)-statistic: \(7.00\)
  - \(p\)-value: \(1.000000\)
- **Effect Size (Cohen's \(d\))**: \(0.00445\) (Negligible effect)
- **Confidence Intervals of Difference**:
  - 95% Analytical CI (t-distribution): \([-0.17103, 0.17226]\)
  - 95% Bootstrap Resampling CI: \([-0.10364, 0.09773]\)
- **Normality Check of Differences (Shapiro-Wilk)**:
  - \(w\)-statistic: \(0.97232\)
  - \(p\)-value: \(0.889921\) (Normality assumption is valid)

### 2.2 Candidate v1.4 vs. Geometry Model v1.2 (Phase 3.1A)
- **Mean Difference**: $+0.00923$
- **Relative Difference**: **-0.60%**
- **Paired t-test**:
  - \(t\)-statistic: \(0.18032\)
  - \(p\)-value: \(0.865670\)

### 2.3 Candidate v1.4 vs. Optimized Baseline v1.1
- **Mean Difference**: $-0.05127$
- **Relative Difference**: **+3.19%** (Improvement in mean error)
- **Paired t-test**:
  - \(t\)-statistic: \(-0.77244\)
  - \(p\)-value: \(0.482960\) (Non-significant)

---

## 3. Computational Profile Matrix

Measured metrics represent average profiles across runs:

| Version | Parameters | Epoch Time (avg) | Checkpoint Size | VRAM Usage | Latency/Sample |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline v1.1** | 1,413,889 | ~15.2s | ~5.6 MB | ~850 MB | ~1.4 ms |
| **Geometry v1.2** | 1,363,281 | ~16.1s | ~5.4 MB | ~880 MB | ~1.6 ms |
| **Attention v1.3**| 1,414,022 | ~17.1s | ~5.7 MB | ~920 MB | ~1.8 ms |
| **IRM v1.4 (Cand)**| 1,455,974 | ~17.9s | ~5.9 MB | ~985 MB | ~2.1 ms |

---

## 4. Promotion & Scientific Conclusions

### 4.1 Promotion Gating Criteria Analysis
1. **RMSE Lower than Phase 3.1B**: **FAIL** (\(1.55349 > 1.55288\))
2. **Paired t-test Significance (\(p < 0.05\))**: **FAIL** (\(p = 0.9925\))
3. **No Benchmark Regressions**: **PASS** (Performance remains bounded with no model collapse)
4. **Computational Footprint**: **PASS** (Latency and parameter overhead are acceptable)

### 4.2 Official Promotion Decision
> [!WARNING]
> **REJECTED FOR PROMOTION**. The IRM module candidate v1.4 does not achieve statistically significant RMSE reduction over the active baseline. Under repository governance rules, **Baseline v1.3 (Geometry Attention) is retained as the active baseline**.
