# Phase 3.1B Promotion Decision
**Experiment ID**: `P3.1B-GEOMETRY-ATTENTION`  
**Certified Date**: 2026-07-15  
**Candidate Version**: Geometry Attention v1.3  
**Active Baseline Reference**: Optimized Baseline v1.1  
**Promotion Decision**: **PROMOTE TO BASELINE v1.3**  

---

## 1. Governance Evaluation Matrix

According to the constitutional Repository Governance Manual, a candidate version is promoted to the new baseline if and only if it satisfies all of the following requirements:

| Evaluation Pillar | Requirement | Candidate Status | Result |
| :--- | :--- | :--- | :---: |
| **1. Prediction Accuracy** | Mean RMSE lower than current baseline ($1.6048$) | **$1.5529$** | **PASS** |
| **2. Statistical Significance** | Paired t-test $p$-value $< 0.05$ across 5 seeds | **$p = 0.01778$** | **PASS** |
| **3. Baseline Preservation** | No modifications permitted to Baseline v1.1 code | Code is untouched | **PASS** |
| **4. Reproducibility Protection** | Cache checksums and random split IDs preserved | Hashing tests passed | **PASS** |
| **5. Parameter Footprint** | Extremely parameter-efficient | **Only +133 parameters** | **PASS** |
| **6. Software Quality** | SOLID and Clean Architecture compliant | Decoupled config & factory | **PASS** |

---

## 2. Decision Rationale

### 2.1 Statistical Validation
The candidate version, Geometry Attention v1.3 (`--model_type geometry_attention`), is the first geometry-aware GNN model to satisfy the strict significance gate. The paired t-test $p$-value of $0.01778$ ($t = -3.8841$) indicates that the improvement over Optimized Baseline v1.1 is highly statistically significant, with a very large effect size (Cohen's $d = -1.7370$).

### 2.2 Architectural Decoupling
Rather than polluting the graph encoders with coordinates as in Phase 3.1A, Phase 3.1B decouples coordinates from node representations and injects spatial distance biases directly inside the cross-attention layer. This preserves the coordinate-free properties of the encoders and keeps the parameters overhead minimal (only 133 weights added, compared to 13,728 added in Phase 3.1A).

---

## 3. Promotion Action Items

1. **Promotion Declaration**: The active baseline of the ProtLigGNN repository is officially upgraded from **Optimized Baseline v1.1** to **Geometry Attention v1.3**.
2. **CLI default**: Keep default `--model_type baseline` as the frozen reference code. All comparative research should invoke `--model_type geometry_attention` for the new active baseline, or `--model_type baseline` for the legacy freeze.
3. **Registry update**: The experiment runs `run_054` to `run_058` are officially promoted inside `experiments/registry.csv`.
