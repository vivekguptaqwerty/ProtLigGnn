import torch
import torch.nn as nn
from typing import Tuple, Optional

class EquivariantInteraction(nn.Module):
    """
    Base interface for all equivariant interaction modules.
    """
    def forward(
        self,
        node_features: torch.Tensor,
        coordinates: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_features: Shape (N, D)
            coordinates: Shape (N, 3)
            edge_index: Shape (2, E)
            edge_attr: Shape (E, D_edge) or None
            
        Returns:
            updated_features: Shape (N, D)
            updated_coordinates: Shape (N, 3)
        """
        raise NotImplementedError
