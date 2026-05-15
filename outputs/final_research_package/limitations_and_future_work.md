# Limitations and Future Work

## Limitations

- ProtLigGNN performance is evaluated computationally; experimental binding validation is not included.
- The SMILES LSTM is lightweight and character-level, so chemical syntax and scaffold quality depend heavily on RDKit filtering after generation.
- AutoDock Vina is used as secondary computational validation, not as ground truth.
- RDKit descriptors and validity checks are filtering tools and are not the novelty of the project.
- Docking boxes are computational approximations and may not capture all induced-fit or solvent effects.

## Future Work

- Train stronger molecular generators such as graph-based or transformer-based models.
- Add protein-conditioned generation rather than target-agnostic SMILES generation.
- Add more rigorous docking box selection around known binding pockets.
- Validate top generated candidates with molecular dynamics or experimental assays.
- Expand the closed-loop GenAI + ProtLigGNN system with active learning from scored candidates.
