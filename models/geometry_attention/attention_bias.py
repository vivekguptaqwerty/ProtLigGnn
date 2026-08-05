import torch
import torch.nn as nn
from models.geometry_attention.interfaces import AttentionBias
from models.geometry.radial_basis import GaussianRadialBasis
from models.geometry_attention.geometry_attention_utils import compute_pairwise_distances

class RBFAttentionBias(AttentionBias):
    """Gaussian RBF-based attention bias implementation."""
    
    def __init__(self, num_basis: int = 32, start: float = 0.0, stop: float = 12.0, num_heads: int = 4) -> None:
        super().__init__()
        self.rbf = GaussianRadialBasis(num_basis=num_basis, start=start, stop=stop)
        self.projector = nn.Linear(num_basis, num_heads)
        
    def forward(self, ligand_pos: torch.Tensor, protein_pos: torch.Tensor) -> torch.Tensor:
        # 1. Compute pairwise distance matrix [L, P]
        dists = compute_pairwise_distances(ligand_pos, protein_pos)
        
        # 2. Expand via RBF [L, P, num_basis]
        rbf_features = self.rbf(dists)
        
        # 3. Project to [L, P, num_heads]
        bias = self.projector(rbf_features)
        
        # 4. Permute to [num_heads, L, P]
        return bias.permute(2, 0, 1)


class LearnedAttentionBias(AttentionBias):
    """Placeholder learned attention bias illustrating Open-Closed Principle compliance."""
    
    def __init__(self, num_heads: int = 4) -> None:
        super().__init__()
        # Simply uses a learnable bias scalar per head
        self.bias_param = nn.Parameter(torch.zeros(num_heads, 1, 1))
        
    def forward(self, ligand_pos: torch.Tensor, protein_pos: torch.Tensor) -> torch.Tensor:
        L = ligand_pos.size(0)
        P = protein_pos.size(0)
        # Returns a static learnable offset broadcasted across L and P
        return self.bias_param.expand(-1, L, P)


def bias_factory(bias_type: str, num_basis: int, start: float, stop: float, num_heads: int) -> AttentionBias:
    """Factory method to return the configured attention bias subclass."""
    if bias_type == "rbf":
        return RBFAttentionBias(num_basis=num_basis, start=start, stop=stop, num_heads=num_heads)
    elif bias_type == "learned":
        return LearnedAttentionBias(num_heads=num_heads)
    else:
        raise ValueError(f"Unknown attention bias type: {bias_type}")
