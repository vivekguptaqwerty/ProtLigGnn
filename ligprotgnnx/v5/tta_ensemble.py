import torch
import torch.nn as nn
import numpy as np

class TestTimeAugmentationEngine:
    """
    Test-Time Augmentation (TTA) & Multi-Conformer Ensemble Engine for LigProtGNN-X v5.0.
    Generates coordinate jitter spatial augmentations during evaluation passes and averages predictions
    to boost Pearson R correlation and uncertainty calibration zero-shot.
    """
    __test__ = False # Exclude from pytest test class collection
    
    def __init__(self, num_augmentations: int = 5, jitter_std: float = 0.05):
        self.num_augmentations = num_augmentations
        self.jitter_std = jitter_std

    def apply_spatial_jitter(self, coords: torch.Tensor) -> torch.Tensor:
        """Applies small isotropic Gaussian 3D coordinate jitter (sigma = 0.05 Angstroms)."""
        noise = torch.randn_like(coords) * self.jitter_std
        return coords + noise

    def predict_with_tta(self, model: nn.Module, batch) -> tuple:
        """
        Executes model inference across K spatial TTA augmentations.
        Returns ensemble mean affinity prediction and calibrated epistemic variance.
        """
        model.eval()
        affinities = []
        uncertainties = []

        with torch.no_grad():
            for k in range(self.num_augmentations):
                # Clone batch and apply coordinate jitter to ligand/protein positions if present
                batch_k = batch.clone()
                if hasattr(batch_k, 'pos_ligand') and batch_k.pos_ligand is not None:
                    batch_k.pos_ligand = self.apply_spatial_jitter(batch_k.pos_ligand)
                if hasattr(batch_k, 'pos_protein') and batch_k.pos_protein is not None:
                    batch_k.pos_protein = self.apply_spatial_jitter(batch_k.pos_protein)

                pred_affinity, pred_uncert = model(batch_k)
                affinities.append(pred_affinity.view(-1))
                uncertainties.append(pred_uncert.view(-1))

        affinities_stack = torch.stack(affinities, dim=0) # [K, B]
        uncertainties_stack = torch.stack(uncertainties, dim=0) # [K, B]

        ensemble_mean = torch.mean(affinities_stack, dim=0)
        ensemble_var = torch.mean(uncertainties_stack, dim=0) + torch.var(affinities_stack, dim=0)

        return ensemble_mean, ensemble_var
