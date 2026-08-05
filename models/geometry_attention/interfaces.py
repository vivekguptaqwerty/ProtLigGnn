import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class AttentionBias(nn.Module, ABC):
    """Abstract base class interface for pluggable attention bias modules."""
    
    @abstractmethod
    def forward(self, ligand_pos: torch.Tensor, protein_pos: torch.Tensor) -> torch.Tensor:
        """Computes the spatial attention bias matrix.
        
        Args:
            ligand_pos: Coordinates tensor of shape [L, 3] for ligand atoms.
            protein_pos: Coordinates tensor of shape [P, 3] for protein residues.
            
        Returns:
            A tensor of shape [num_heads, L, P] representing the additive bias.
        """
        pass
Class = AttentionBias
