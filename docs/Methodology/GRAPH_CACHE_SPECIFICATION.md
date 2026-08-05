# Graph Cache Specification

This document defines the schema, serialization protocol, and validation checks for the persistent preprocessed graph dataset cache of ProtLigGNN.

---

## 1. Serialization Protocol & Target File

- **Cache Filename:** `data/pdbbind2020/processed_dataset.pt`
- **Serialization Library:** `torch.save` (leveraging PyTorch's tensor-optimized serialization with pickle fallback for primitives).
- **Structure:** A top-level Python dictionary containing both the cache manifest metadata and the pre-constructed list of graph samples.

---

## 2. Cache Schema

The serialized cache file contains a dictionary with the following schema:

```json
{
  "manifest": {
    "cache_version": "1.0",
    "creation_date_utc": "2026-07-11T17:45:00Z",
    "dataset_hash": "5735082df2414002f80ca63192bec62e5aacf646aff92ffd1b605394506379c7",
    "config_hash": "e5c4ffbb089b55547d9f117a09479b3cddbc7e5ae91eeaae0b69c2f42f30971c",
    "feature_version": "1.0",
    "graph_schema_version": "1.0",
    "ligand_feature_dim": 9,
    "protein_feature_dim": 30,
    "pdb_count": 1000,
    "environment_version": "ProtLigGNN Environment v1.0",
    "pytorch_version": "2.3.1+cu121",
    "pyg_version": "2.8.0"
  },
  "samples": [
    [
      "ligand_graph_data_obj",   // torch_geometric.data.Data
      "protein_graph_data_obj",  // torch_geometric.data.Data
      "affinity_label",          // float
      "pdb_id"                   // string
    ],
    ...
  ]
}
```

---

## 3. Validation Checklist

Before a cached dataset is loaded, it must pass the following integrity checks. If ANY check fails, the cache is invalidated and automatically rebuilt:

1. **Dataset Fingerprint Match:** The current dataset files hash must match the `dataset_hash` stored in the manifest.
2. **Configuration Fingerprint Match:** The active benchmark hyperparameters must match the `config_hash` stored in the manifest.
3. **Dimension Consistency:**
   - Ligand graph node feature size must match `ligand_feature_dim` (9).
   - Protein graph node feature size must match `protein_feature_dim` (30).
4. **PDB Sample Count Verification:** The number of samples stored in `samples` must match the expected size (`max_samples` or total index file rows).
5. **Data Structure Validation:** Verify that the first sample contains a tuple of length 4, and the graph elements are instances of `torch_geometric.data.Data`.
6. **Software Compatibility Check:** Verify that the runtime PyTorch and PyG versions match the serialization environment to prevent ABI conflicts.
