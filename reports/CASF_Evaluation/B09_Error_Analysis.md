# B09 Error Analysis: CASF-2016

## Localized Failures
- **Largest Errors**: Complexes with very high affinity (pK$_d$ > 10.0) or very low affinity (pK$_d$ < 3.0) show the largest absolute errors. The model underpredicts high affinities and overpredicts low affinities (regression to the training mean).
- **Structure Anomalies**: SDF sanitization failures (valency errors resolved by MOL2 fallback) indicate that ligand structural quality varies, introducing noise in feature extraction.
