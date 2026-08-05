import torch
import torch.nn as nn
from .radial_basis import GaussianRadialBasis

class DistanceEncoder(nn.Module):
    """Encodes continuous distances into projected high-dimensional edge features.
    
    Combines Gaussian RBF mapping with a learnable linear layer to produce
    the final edge representations of the specified dimensionality.
    """
    
    def __init__(self, num_basis: int = 32, start: float = 0.0, stop: float = 12.0, projection_dim: int = 32) -> None:
        """Initializes the DistanceEncoder.
        
        Args:
            num_basis: Number of Gaussian radial basis functions.
            start: Start distance of the range in Angstroms.
            stop: End distance of the range in Angstroms.
            projection_dim: Dimensionality of the projected edge attribute vector.
        """
        super().__init__()
        self.rbf = GaussianRadialBasis(num_basis, start, stop)
        self.projector = nn.Linear(num_basis, projection_dim)
        
    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """Encodes continuous distances.
        
        Args:
            dist: A tensor of shape [num_edges] containing continuous distances.
            
        Returns:
            A tensor of shape [num_edges, projection_dim] containing projected edge attributes.
        """
        rbf_mapped = self.rbf(dist)
        return self.projector(rbf_mapped)
