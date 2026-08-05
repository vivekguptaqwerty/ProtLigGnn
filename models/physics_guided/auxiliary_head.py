import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class AuxiliaryHead(nn.Module, ABC):
    """Abstract base class defining the contract for auxiliary multitask prediction heads."""
    
    @abstractmethod
    def forward(self, ligand_nodes: torch.Tensor, protein_nodes: torch.Tensor) -> torch.Tensor:
        """Executes the forward pass to generate auxiliary target predictions."""
        pass
