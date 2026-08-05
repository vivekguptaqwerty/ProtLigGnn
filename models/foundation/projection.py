import torch
import torch.nn as nn

class ProjectionMLP(nn.Module):
    """
    Multi-layer projection network to adapt pretrained foundation embeddings.
    
    Architecture:
        Foundation Embedding -> Linear -> LayerNorm -> GELU -> Dropout -> Linear -> Projected Embedding
    """
    
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projects input embeddings of dimension input_dim to output_dim.
        
        Args:
            x: Input tensor of shape (..., input_dim)
            
        Returns:
            Projected tensor of shape (..., output_dim)
        """
        return self.mlp(x)
