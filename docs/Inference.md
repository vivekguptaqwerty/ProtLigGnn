# Unified Inference API

All predictions utilize `model.predict(ligand_batch, protein_batch)` returning a structured `PredictionResult` carrying:
- `affinity`
- `confidence`
- `prediction_interval`
- `uncertainty`
- `explanation`
- `robustness`
- `ood_probability`
- `metadata`