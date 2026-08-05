# Constituent Certificate of Phase 3.1B Scientific Evaluation
**Certificate ID**: `PLGNN-CERT-20260715-V1.3`  
**Experiment ID**: `P3.1B-GEOMETRY-ATTENTION`  
**Certified Date**: 2026-07-15  
**Active Baseline Upgrade**: Geometry Attention v1.3 (`--model_type geometry_attention`)  

---

## 1. Metadata and Context
- **Candidate Model**: ProtLigGNNGeometryAttention (Distance-Biased Cross-Attention GNN)
- **Reference Model**: Optimized Baseline v1.1
- **Developer/Auditor**: Deep learning Researcher & Auditor
- **Hardware Platform**: NVIDIA GPU CUDA 12.1 Enabled (GeForce RTX 3050 Laptop GPU)

---

## 2. Dataset and Environment Fingerprints
- **Dataset Cache File**: `data/pdbbind2020/processed_dataset.pt`
- **Cache SHA-256 Checksum**: `6fb4385c2136b197ef95501c4eea8775ad485702bfa5391b8cdbda909a108a3f`
- **Benchmark Seeds**: `[42, 123, 777, 2024, 3407]`
- **Python Version**: `3.11.9`
- **PyTorch Version**: `2.3.1+cu121`
- **PyTorch Geometric (PyG) Version**: `2.5.3`

---

## 3. Certified Performance & Significance

### 3.1 Aggregated Metrics Comparison (Test Set)

| Version | Mean RMSE | Mean MAE | Mean PCC | Mean Spearman | Mean $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Optimized Baseline v1.1** | $1.6048 \pm 0.0520$ | $1.2685 \pm 0.0301$ | $0.5064 \pm 0.0484$ | $0.4911 \pm 0.0405$ | $0.2006 \pm 0.0791$ |
| **Geometry Attention v1.3** | **$1.5529 \pm 0.0727$** | **$1.2153 \pm 0.0329$** | **$0.5376 \pm 0.0341$** | **$0.5303 \pm 0.0364$** | **$0.2531 \pm 0.0679$** |

### 3.2 Paired t-Test and Effect Size
- **RMSE reduction**: $-0.0519$ ($t = -3.88$, **$p = 0.01778 < 0.05$**)
- **MAE reduction**: $-0.0532$ ($t = -5.12$, **$p = 0.00687 < 0.01$**)
- **Cohen's d (RMSE)**: $-1.7370$ (Very large effect size)
- **95% Bootstrap CI of Difference**: $[-0.0721, -0.0274]$

---

## 4. Scientific Conclusions & Limitations

1. **Successful controlled validation**: Directly biasing dot-product attention logits with explicit physical Euclidean distances yields a statistically significant reduction in binding affinity prediction error ($p = 0.01778$).
2. **Weak spatial attention correlation**: The model's attention weights do not align with physical close contacts (mean Pearson $r = -0.0340$), indicating that GNNs route information based on topological node features rather than physical contact proximity under regression supervision.
3. **High parameter efficiency**: By keeping the encoders coordinate-free, the model achieves performance equivalent to Phase 3.1A while adding only **133 extra parameters**.

---

## 5. Official Promotion Decision

### **STATUS: PROMOTE TO BASELINE v1.3**

*Certified by:*  
**Principal AI Research Scientist & Benchmark Committee Chair**  
*Date: 2026-07-15*
