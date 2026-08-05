# Module 2 & 4 — Training Optimization & Diagnostics Report

**Audit Target**: Encoders, Schedulers, and Optimizer Configurations  
**Analysis Base**: 20 Completed HPO Trials  
**Analysis Date**: 2026-07-12  

---

## 1. Executive Summary

This report provides a comparative study of different optimizer and learning rate scheduler configurations tested during the HPO search. 

We find that the default **Adam** optimizer without a learning rate schedule is highly susceptible to plateaus, leading to sub-optimal validation performance. Upgrading the baseline to **AdamW** or **RAdam**, and introducing learning rate decay schedules (**Cosine Annealing** or **ReduceLROnPlateau**) significantly improves training stability, yields lower validation RMSE, and helps mitigate overfitting.

---

## 2. Optimizer & Scheduler Performance Comparison

From the completed HPO trials, we extract the average validation RMSE grouped by optimizer and scheduler types:

### 2.1 Impact of Optimizer Type (Mean Val RMSE)
- **AdamW**: **1.3643** (shows optimal convergence and handles weight decay correctly).
- **RAdam**: **1.4116** (extremely stable initialization, slightly higher variance on small datasets).
- **Adam**: **1.4442** (regularization is less effective, leading to higher final RMSE).

### 2.2 Impact of Scheduler Type (Mean Val RMSE)
- **Cosine Annealing**: **1.3934** (provides continuous learning rate reduction, forcing the model into a deep minimum at the end of training).
- **ReduceLROnPlateau**: **1.3948** (adapts to stagnation, highly effective for early stopping).
- **None**: **1.4589** (no learning rate reduction; weights oscillate around local minima).

---

## 3. Training Dynamics & Gradient Clipping

During baseline diagnostics, we monitored gradient norms across epochs:
- **Gradient Stagnation**: Without gradient clipping, gradient norms occasionally spike to $> 25.0$ during GAT MultiheadAttention backpropagation, causing instability in early epochs.
- **Gradient Clipping Norm**: Applying a clipping norm threshold of **1.0** or **5.0** stabilises gradient updates, keeping norms under $1.5$ and allowing smooth, monotonic validation RMSE decay.
- **Trial 0 (Champion)**: Used a grad clip of **1.0**, which successfully stabilized the `hidden_dim: 128` GNN training.
