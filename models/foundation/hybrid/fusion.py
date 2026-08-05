import torch
import torch.nn as nn
import torch.nn.functional as F

class ConcatenationFusion(nn.Module):
    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.linear = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, protein_rep: torch.Tensor, ligand_rep: torch.Tensor) -> torch.Tensor:
        # Concatenate along the last dimension
        concat = torch.cat([protein_rep, ligand_rep], dim=-1)
        return self.norm(F.gelu(self.linear(concat)))


class WeightedSumFusion(nn.Module):
    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.norm = nn.LayerNorm(dim)

    def forward(self, protein_rep: torch.Tensor, ligand_rep: torch.Tensor) -> torch.Tensor:
        return self.norm(self.alpha * protein_rep + self.beta * ligand_rep)


class GatedFusion(nn.Module):
    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.gate_linear = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, protein_rep: torch.Tensor, ligand_rep: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([protein_rep, ligand_rep], dim=-1)
        gate = torch.sigmoid(self.gate_linear(concat))
        fused = gate * protein_rep + (1.0 - gate) * ligand_rep
        return self.norm(fused)


class ResidualFusion(nn.Module):
    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, protein_rep: torch.Tensor, ligand_rep: torch.Tensor) -> torch.Tensor:
        # Basic sum
        base_sum = protein_rep + ligand_rep
        # Residual adapter
        residual = self.mlp(base_sum)
        return self.norm(base_sum + residual)
