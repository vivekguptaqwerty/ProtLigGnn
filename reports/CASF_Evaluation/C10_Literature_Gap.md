# C10 Literature Gap: CASF-2016

## SOTA Feature Checklist
- **Physics-Guided Loss**: Models like PIGNet model steric clashes and coordinate interaction potentials directly in the loss function (missing in our MSE loss).
- **Equivariant Convolutions**: Models like EGNN or EquiBind preserve 3D coordinate equivariance under rotations (our model uses coordinates inside GAT, which is invariant but less structurally expressive).
- **Scale-Free Distance RBFs**: Using learnable Radial Basis Functions (RBFs) to map distance vectors (our GAT uses raw coordinate differences).
