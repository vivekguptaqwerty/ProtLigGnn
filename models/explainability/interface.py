import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

@dataclass
class ExplanationResult:
    affinity: torch.Tensor
    confidence: torch.Tensor
    residue_scores: torch.Tensor
    atom_scores: torch.Tensor
    residue_rankings: List[int]
    atom_rankings: List[int]
    interaction_heatmap: torch.Tensor
    explanation_method: str
    explanation_confidence: torch.Tensor  # A calibrated combination of stability, faithfulness, and agreement

class ExplanationEngine(nn.Module):
    """
    Unified abstract interface for explainability methods.
    """
    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.base_model = base_model

    def explain(
        self,
        ligand_batch: Any,
        protein_batch: Any
    ) -> ExplanationResult:
        raise NotImplementedError("Subclasses must implement the explain method.")
