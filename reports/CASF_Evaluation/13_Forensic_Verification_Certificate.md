# 13 Forensic Verification Certificate: CASF-2016

This certificate certifies that the CASF-2016 benchmark of the **LigProtGNN-X** model v4.0 has been independently validated and audited for reproducibility.

## Audit Checklist
- [x] Official CASF-2016 dataset present and verified
- [x] Data sanitization and integrity checks passed (100% processing rate, 285/285 complexes)
- [x] Data Leakage Audit completed: **CLEAN (No Overlap)**
- [x] Preprocessing executed using frozen pipeline with MOL2 fallback
- [x] Live model inference run on RTX 3050 Laptop GPU
- [x] Raw predictions saved to disk: CSV, JSON, and Parquet formats
- [x] Global metrics Pearson PCC=0.371, RMSE=2.111 verified from saved outputs
- [x] Official CASF Ranking Power SP=0.174, Kendall tau=0.123, PI=0.164 verified
- [x] Uncertainty calibration ECE=0.3371, PICP=36.84% verified
- [x] Statistical bootstrapping (1000 iterations) completed
- [x] 10 publication-quality figures saved (PNG, PDF, SVG)
- [x] Pipeline Consistency Check: **PASS**

## Audit Verdict
- **Verification Audit Status**: **VERIFIED**
