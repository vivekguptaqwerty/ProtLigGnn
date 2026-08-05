import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict
from dataclasses import dataclass

@dataclass(frozen=True)
class InteractionContext:
    """Canonical immutable data container inside the Interaction Representation Module (IRM)."""
    version: str
    
    # Coordinates (float32)
    ligand_pos: Tensor         # Shape: [L, 3]
    protein_pos: Tensor        # Shape: [P, 3]
    
    # Raw node features (float32)
    ligand_x_raw: Tensor       # Shape: [L, 78]
    protein_x_raw: Tensor      # Shape: [P, 30]
    
    # Encoded representations (float32)
    ligand_x_enc: Tensor       # Shape: [L, hidden_dim]
    protein_x_enc: Tensor      # Shape: [P, hidden_dim]
    
    # Batches / indices (int64)
    ligand_batch: Tensor       # Shape: [L]
    protein_batch: Tensor      # Shape: [P]


class FeatureEncoder(nn.Module):
    """Abstract base class for all pairwise feature encoders in IRM."""
    def forward(self, context: InteractionContext) -> Tensor:
        raise NotImplementedError


class InteractionFusion(nn.Module):
    """Abstract base class for fusing branch embeddings in IRM."""
    def forward(self, features: Dict[str, Tensor]) -> Tensor:
        raise NotImplementedError
