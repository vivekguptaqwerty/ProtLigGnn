# Phase 3.1B Implementation Plan: Distance-Biased Cross-Attention
**Experiment ID**: `P3.1B-GEOMETRY-ATTENTION`  
**Certified Date**: 2026-07-15  
**Status**: Pending Review  

---

## 1. Goal Description
The objective of Phase 3.1B is to evaluate whether injecting continuous physical 3D distance information directly into the cross-attention logits improves protein-ligand interaction modeling and overall binding affinity prediction.

---

## 2. Pluggable Abstraction & Configuration Layer
To adhere to SOLID principles and prevent coupling, we define:
- `AttentionBias`: Abstract base class (`nn.Module`) defining `forward(ligand_pos, protein_pos) -> bias_tensor`.
- `RBFAttentionBias`: Maps continuous distances via Gaussian RBF and linear projections.
- `AttentionBiasConfig`: Configuration dataclass containing all hyperparameters (bins, cutoffs, head count).

---

## 3. Explicit File Structure

```text
models/
└── geometry_attention/
    ├── __init__.py
    ├── interfaces.py              # Abstract interface (AttentionBias)
    ├── config.py                  # AttentionBiasConfig dataclass
    ├── attention_bias.py          # Concrete bias modules (RBFAttentionBias)
    ├── geometry_attention_model.py# Upgraded model classes
    └── geometry_attention_utils.py# Coordinate & distance utility calculations

tests/
├── test_geometry_attention.py            # Basic flow & shape checks
├── test_attention_bias.py                # RBF & learnable mappings
├── test_localization_metrics.py          # Entropy & spatial correlation
└── test_geometry_attention_checkpoint.py # Checkpoint save/load & log_alpha checks
```

---

## 4. Benchmark Artifacts
Every completed seed run folder under `experiments/run_xxx/` will output:
- `geometry_attention_metrics.json`: Per-run metrics.
- `geometry_attention_predictions.csv`: Affinity predictions.
- `interaction_localization_metrics.json`: Pairwise distance/attention scores.
- `interaction_localization_report.md`: Markdown summary of localization.
- `computational_profile.json`: Speed, memory, and FLOPS.
- `attention_head_statistics.json`: Head-wise details.
- `attention_entropy.json`: Attention distribution entropy.
- `distance_bias_statistics.json`: Output range parameters of the bias.

---

## 5. Experiment Registry Metadata
We extend `experiments/registry.csv` with these key interaction fields:
- `model_type`: E.g. `geometry_attention`
- `bias_type`: E.g. `rbf`
- `num_rbf`: Number of basis functions (32)
- `alpha_init`: Initial scale parameter (0.1)
- `alpha_final`: Learned scale parameter at checkpoint
- `localization_score`: Inverse distance correlation
- `attention_entropy`: Normalized attention entropy
- `runtime_overhead`: Overhead vs baseline (%)
- `memory_overhead`: Peak memory overhead vs baseline (%)

---

## 6. Benchmark Stopping Rules
The benchmark script must abort training for a run if any of the following are met:
1. **NaN Loss**: Loss values become `NaN` or `Inf`.
2. **Exploding Gradients**: Gradient norm exceeds $10.0$ for three consecutive steps.
3. **Entropy Collapse**: Attention entropy collapses to $< 0.005$ across all heads.
4. **Uncontrolled Scale**: The learnable parameter $\alpha$ grows beyond $10.0$ (indicating divergence).
5. **Undefined Localization**: Correlation values become `NaN` due to zero variance in attention matrices.

---

## 7. Future Compatibility Notes
The pluggable interfaces are designed to support seamless extensions in future phases, including:
- **Protein distance bias**: Residue-residue distance corrections.
- **Full geometry bias**: All-atom distance representations.
- **Learned positional bias**: Transferred embedding-based positional encodings.
- **Physics-informed interaction priors**: Incorporating Lennard-Jones or electrostatic potentials.
- **Orientation-aware attention**: Angles and directions between nodes.
- **SE(3)-equivariant interaction modules**: Invariant attention over SE(3) representations.

---

## 8. Phase 3.1B Design Review Summary
- **Why the architecture is scientifically controlled**: The model differs from `Optimized Baseline v1.1` by exactly one structural modification: the distance-bias layer in cross-attention. Node features, initializations, and training settings are kept frozen.
- **How reproducibility is preserved**: Pre-processed dataset cache hashes and seeds are verified by the existing protection tests before benchmark execution.
- **Why the software design is extensible**: Future physical (e.g. Lennard-Jones), learned, or contact-based bias formulations can be plugged in by inheriting from `AttentionBias` and updating the configuration.
- **Preparation for future phases**: Decoupling the interaction representation from coordinates preparation allows seamless integration of equivariant geometric descriptors or orientation features in later stages.
