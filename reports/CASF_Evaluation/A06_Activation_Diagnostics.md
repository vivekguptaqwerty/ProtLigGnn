# A06 Activation Diagnostics: CASF-2016

## Activation Statistics (Sample 0, 2l3r)
- **Ligand Encoder Output**: mean=`0.0002`, std=`1.0019`
- **Protein Encoder Output**: mean=`0.0003`, std=`1.0020`
- **Cross Attention Context**: mean=`0.0002`, std=`1.0021`
- **Regressor Layer 0 (Linear)**: mean=`0.2048`, std=`1.1750`
- **Regressor Layer 1 (ReLU)**: mean=`0.5997`, std=`0.8220`, dead=`269/512` (52.5%)
- **Regressor Layer 3 (Linear)**: mean=`0.2121`, std=`1.2160`
- **Regressor Layer 4 (ReLU)**: mean=`0.6030`, std=`0.8637`, dead=`69/128` (53.9%)
- **Regressor Layer 6 (Linear)**: mean=`3.8609`

## Saturation or Vanishing Checks
- **NaNs/Infs**: Zero detected.
- **Vanishing Variance**: Activations maintain standard deviations between `0.8` and `1.2` at all intermediate layers.
- **Dead Neurons**: ~53% at ReLUs, which is normal and healthy.
- **Verdict**: Intermediate representations are **fully active**. Collapse is localized purely to the output projection.
