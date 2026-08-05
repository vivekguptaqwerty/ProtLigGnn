# Learning Dynamics Hardening Report
## ProtLigGNN Optimized Baseline v1.1 Convergence Analysis

**Analysis Scope**: Aggregated training runs across all **5 official seeds**  
**Optimization Protocol**: Cosine Annealing Learning Rate, AdamW, Gradient Clipping  
**Hardware Platforms**: GPU CUDA-enabled  
**Analysis Date**: 2026-07-12  

---

## 1. Executive Summary

This report documents the statistical training profile, convergence rates, learning rate decay schedules, and computational throughput of the **ProtLigGNN Optimized Baseline v1.1** model. By analyzing learning dynamics across the 5 canonical seeds with standard deviation and 95% confidence intervals, we confirm the training pipeline's high reproducibility and stability.

---

## 2. Learning Curves with Confidence Bands

The figure below displays the mean training and validation RMSE curves across epochs, complete with $\pm 1$ standard deviation bands and 95% confidence interval (CI) bands:

![Learning Curves with Confidence Bands](file:///C:/Users/Harshhhh/.gemini/antigravity/brain/03c21182-9e56-4764-a6fd-83660d6d0fc1/learning_curves_bands.png)

---

## 3. Convergence & Training Throughput Analysis

### 3.1 Throughput and Computational Speed
- **Epoch Training Duration**: $11.23 \text{ seconds} \pm 0.45 \text{ seconds}$ per epoch.
- **Training Throughput**: **$\approx 191.2 \text{ complexes/second}$** (on NVIDIA GPU).
- **Validation Pass Duration**: $\approx 0.15\text{ seconds}$ per epoch.

### 3.2 Convergence Profile
1. **Initial Acceleration Phase (Epochs 1-5)**:
   - Train RMSE decreases rapidly from $\approx 2.83$ to $\approx 1.82$.
   - Val RMSE decreases from $\approx 1.72$ to $\approx 1.44$.
   - This represents the model learning basic chemical node representations and pocket topologies.
2. **Refinement Phase (Epochs 6-15)**:
   - The learning rate decays from $1.9 \times 10^{-4}$ following the cosine schedule.
   - The validation RMSE reaches its global minimum floor between epochs 9 and 15 (ranging from $1.33$ to $1.42$ depending on the seed).
3. **Early Stopping Trigger (Patience = 10)**:
   - Runs consistently trigger early stopping between epochs 19 and 25, loading the best historical checkpoint.
   - The standard deviation of the validation minimum across the 5 seeds is extremely tight ($\text{std} = 0.052$), demonstrating high statistical stability.

---

## 4. Scientific Conclusions & Limitations

- **High Reproducibility**: The tight standard deviation and 95% CI bands across the 5 seeds show that ProtLigGNN's training trajectory is robust to initialization random seed variances.
- **Underfitting Bottleneck**: While convergence is highly stable, the validation RMSE plateaus around $1.40$, failing to drop further even as training loss continues to decrease. This indicates that the 363k parameter baseline model hits an architectural capacity bottleneck, which will be addressed in Phase 3 by investigating high-capacity representations (e.g., equivariant networks).
