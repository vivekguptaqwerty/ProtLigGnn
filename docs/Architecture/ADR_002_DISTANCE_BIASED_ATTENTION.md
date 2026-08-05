# ADR 002: Distance-Biased Cross-Attention Abstraction and Scaling (Revised)
**Status**: Proposed  
**Date**: 2026-07-15  

---

## 1. Context and Problem Statement
Phase 3.1A proved that adding explicit ligand edge distance features in the ligand encoder does **not** automatically translate to physically realistic spatial cross-attention. The correlation between attention weights and pairwise physical coordinates remained extremely low ($r \le 0.16$).

To resolve this bottleneck, we must directly inject spatial distance information into the cross-attention calculation. However, the system must remain:
1. Modular and pluggable (to allow future physical, learned, or contact-based bias formulations).
2. Configuration-driven (preventing hardcoded constants in the model definition).
3. Mathematically controlled (preventing attention logit saturation and vanishing gradients).

---

## 2. Decision
1. **Abstract Interface**: Define `AttentionBias` as an abstract base class inheriting from `nn.Module` to decouple the attention logic from the RBF mapping.
2. **Configuration Class**: Implement `AttentionBiasConfig` to encapsulate RBF bins, cutoffs, head dimensions, and scaling options.
3. **Logarithmic Scaling Parameter**: Implement the learnable scaling parameter $\alpha$ in log-space:
   $$\alpha = \exp(\log\_alpha)$$
   Initialize `log_alpha = nn.Parameter(torch.tensor(math.log(0.1)))`. This guarantees $\alpha > 0$, ensures smooth gradient optimization, and prevents sign inversion.
4. **Additive Logit Scaling**:
   $$\text{AttentionWeights} = \text{Softmax}(S + \alpha B)$$
5. **Interaction Localization Suite**: Build programmatics to evaluate attention-distance correlation (Pearson, Spearman) and attention entropy as first-class validation metrics.

---

## 3. Alternatives Considered

### Alternative A: Coordinate Concatenation
- **Description**: Concatenate coordinate features $\vec{p}$ directly to node representations before cross-attention.
- **Reason for Rejection**: This violates the Open-Closed Principle, increases the input feature dimensionality, and changes the model capacity, which confounds the comparison against Baseline v1.1.

### Alternative B: Fixed Scaling ($\alpha = \text{constant}$)
- **Description**: Add distance bias with a fixed scale (e.g. $\alpha = 1.0$).
- **Reason for Rejection**: A fixed scale does not allow the model to learn the relative importance of physical geometry versus topological node representations, and could lead to logit saturation if the initial scale is too large.

---

## 4. Risks and Expected Limitations
- **Overfitting Risk**: Adding the projection layers ($\text{Linear}(K, H)$) could lead to overfitting on the small dataset. We mitigate this by keeping the projection dimensions small ($32 \to 4$) and regularizing using dropout.
- **Logit Saturation**: Large RBF outputs could still dominate attention logits if $\alpha$ explodes. We mitigate this by initializing $\alpha = 0.1$ and supporting `normalize_bias = True` (applying LayerNorm to the bias).
