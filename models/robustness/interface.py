import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class RobustnessResult:
    affinity: torch.Tensor
    confidence: torch.Tensor
    explanation: Any  # ExplanationResult object
    perturbation_type: str
    perturbation_level: float
    rmse: float
    uncertainty: float
    robustness_score: float  # Composite Robustness Index (CRI)
    calibration: float  # ECE score under perturbation
    explanation_similarity: float  # Spearman rank correlation vs. original explanation

class RobustnessEvaluator(nn.Module):
    """
    Unified abstract interface for robustness evaluations.
    """
    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.base_model = base_model

    def evaluate(
        self,
        ligand_batch: Any,
        protein_batch: Any
    ) -> RobustnessResult:
        raise NotImplementedError("Subclasses must implement the evaluate method.")
