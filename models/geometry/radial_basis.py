import torch
import torch.nn as nn

class GaussianRadialBasis(nn.Module):
    """Gaussian Radial Basis Function expansion layer.
    
    Maps a continuous distance tensor to a high-dimensional vector space using
    a bank of Gaussian kernels with evenly spaced centers.
    """
    
    def __init__(self, num_basis: int = 32, start: float = 0.0, stop: float = 12.0) -> None:
        """Initializes the GaussianRadialBasis.
        
        Args:
            num_basis: Number of basis functions (Gaussian centers).
            start: Start distance of the range in Angstroms (typically 0.0).
            stop: End distance of the range in Angstroms (typically 12.0).
        """
        super().__init__()
        self.num_basis = num_basis
        self.start = start
        self.stop = stop
        
        # Calculate evenly spaced centers (means) for the Gaussians
        means = torch.linspace(start, stop, num_basis)
        self.register_buffer("means", means)
        
        # Calculate spacing between centers to compute the bandwidth (gamma)
        spacing = (stop - start) / (num_basis - 1)
        self.gamma = 1.0 / (2.0 * (spacing ** 2))
        
    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """Applies the Gaussian RBF mapping to a distance tensor.
        
        Args:
            dist: A tensor of shape [num_edges] containing continuous distances.
            
        Returns:
            A tensor of shape [num_edges, num_basis] containing mapped RBF features.
        """
        # Expand dimensions to broadcast against the means
        dist = dist.unsqueeze(-1)  # [num_edges, 1]
        
        # Compute squared differences: (d - mu)^2
        diff = dist - self.means  # [num_edges, num_basis]
        
        # Apply Gaussian: exp(-gamma * (d - mu)^2)
        return torch.exp(-self.gamma * (diff ** 2))
