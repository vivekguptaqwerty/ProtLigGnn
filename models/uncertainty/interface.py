import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple

@dataclass
class PredictionWithUncertainty:
    affinity: torch.Tensor                # predicted binding affinity mean (shape: [Batch])
    aleatoric_uncertainty: torch.Tensor   # data noise variance (shape: [Batch])
    epistemic_uncertainty: torch.Tensor   # model parameter uncertainty (shape: [Batch])
    total_uncertainty: torch.Tensor       # total variance (aleatoric + epistemic) (shape: [Batch])
    confidence: torch.Tensor              # reliability/confidence score [0, 1] (shape: [Batch])
    prediction_interval: Tuple[torch.Tensor, torch.Tensor]  # 95% interval (lower, upper)
    method: str                           # "mc_dropout", "ensemble", "evidential"
    calibrated: bool                      # True if scaling post-processing was applied

class UncertaintyEstimator(nn.Module):
    """
    Common abstract base class interface for all predictive uncertainty estimation methods.
    """
    def predict_with_uncertainty(
        self,
        ligand_batch,
        protein_batch
    ) -> PredictionWithUncertainty:
        raise NotImplementedError("Subclasses must implement predict_with_uncertainty")
