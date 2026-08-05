import torch
import torch.nn as nn
from models.physics_guided.auxiliary_head import AuxiliaryHead

class ContactPredictionHead(AuxiliaryHead):
    """Concrete prediction head to map pairwise node embeddings to contact map logits."""
    
    def __init__(self, node_dim: int, hidden_dim: int, dropout: float = 0.1, 
                 activation: str = "relu", normalization: str = "layer") -> None:
        super().__init__()
        
        # Concat of Query (Ligand) and Key (Protein) node embeddings
        in_dim = node_dim * 2
        
        if activation == "gelu":
            act_fn = nn.GELU()
        else:
            act_fn = nn.ReLU()
            
        if normalization == "layer":
            norm_layer = nn.LayerNorm(hidden_dim)
        else:
            norm_layer = nn.Identity()
            
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            norm_layer,
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, ligand_nodes: torch.Tensor, protein_nodes: torch.Tensor) -> torch.Tensor:
        L = ligand_nodes.size(0)
        P = protein_nodes.size(0)
        
        # Broad-cast concatenation
        ligand_expanded = ligand_nodes.unsqueeze(1).expand(-1, P, -1)
        protein_expanded = protein_nodes.unsqueeze(0).expand(L, -1, -1)
        
        pairwise = torch.cat([ligand_expanded, protein_expanded], dim=-1)  # [L, P, 2 * D]
        logits = self.mlp(pairwise).squeeze(-1)  # [L, P]
        return logits
