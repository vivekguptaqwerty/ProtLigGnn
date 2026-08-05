import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from models.equivariant.interface import EquivariantInteraction
from models.equivariant.config import EquivariantConfig

class EGNNLayer(nn.Module):
    """
    Equivariant Graph Neural Network (EGNN) Message Passing Layer.
    Translation-equivariant, rotation-equivariant, and permutation-invariant.
    
    Key equivariance property:
    - Messages depend only on node features h_i, h_j and squared distance ||x_i - x_j||^2
      (all SE(3)-invariant quantities).
    - Coordinate updates use the direction vector (x_i - x_j) scaled by an invariant scalar,
      preserving SE(3)-equivariance of the coordinate output.
    """
    def __init__(
        self, 
        node_dim: int, 
        edge_dim: int = 0, 
        coord_updates: bool = True, 
        residual: bool = True, 
        layer_norm: bool = True,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.coord_updates = coord_updates
        self.residual = residual
        
        # Message MLP: input dimension is [h_i, h_j, dist_sq, edge_attr]
        in_dim = node_dim * 2 + 1 + edge_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim, node_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, node_dim)
        )
        
        # Coordinate MLP: maps message to coordinate scaling factor (scalar)
        if coord_updates:
            self.coord_mlp = nn.Sequential(
                nn.Linear(node_dim, node_dim),
                nn.GELU(),
                nn.Linear(node_dim, 1, bias=False)  # No bias to preserve pure equivariance
            )
            # Initialize coord MLP to near-zero outputs to prevent early coordinate drift
            nn.init.xavier_uniform_(self.coord_mlp[0].weight, gain=0.001)
            nn.init.xavier_uniform_(self.coord_mlp[2].weight, gain=0.001)
            
        # Node feature update MLP: input is [h_i, aggregated message]
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, node_dim)
        )
        
        self.norm = nn.LayerNorm(node_dim) if layer_norm else nn.Identity()

    def forward(
        self, 
        h: torch.Tensor, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_attr: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        row, col = edge_index
        
        # 1. Compute squared Euclidean distance (SE(3)-invariant scalar)
        coord_diff = x[row] - x[col]  # Direction vector (equivariant)
        dist_sq = torch.sum(coord_diff ** 2, dim=-1, keepdim=True)  # Invariant scalar
        
        # 2. Compute messages from invariant inputs only
        edge_inputs = [h[row], h[col], dist_sq]
        if edge_attr is not None:
            edge_inputs.append(edge_attr)
        edge_input = torch.cat(edge_inputs, dim=-1)
        m_ij = self.edge_mlp(edge_input)
        
        # 3. Update Coordinates (equivariant: direction * invariant_scalar)
        if self.coord_updates:
            scale = self.coord_mlp(m_ij)  # Invariant scalar per edge
            trans = coord_diff * scale     # Equivariant: direction * scalar
            
            # Clamp coordinate update magnitudes for strict numerical stability
            trans = torch.clamp(trans, min=-10.0, max=10.0)
            
            x_update = torch.zeros_like(x)
            x_update.scatter_add_(0, row.unsqueeze(-1).expand(-1, 3), trans)
            x = x + x_update
            
        # 4. Aggregate messages (invariant under SE(3))
        m_i = torch.zeros((h.size(0), h.size(1)), device=h.device, dtype=h.dtype)
        m_i.scatter_add_(0, row.unsqueeze(-1).expand(-1, h.size(1)), m_ij)
        
        # 5. Update node features (invariant)
        node_inputs = torch.cat([h, m_i], dim=-1)
        h_new = self.node_mlp(node_inputs)
        
        if self.residual:
            h = h + h_new
        else:
            h = h_new
            
        return self.norm(h), x


class EGNN(EquivariantInteraction):
    """
    Multi-layer EGNN implementing EquivariantInteraction interface.
    Optionally projects input features to internal hidden_dim before message passing.
    """
    def __init__(self, config: EquivariantConfig, input_dim: Optional[int] = None) -> None:
        super().__init__()
        self.config = config
        
        # Optional input projection: maps GNN output dim -> EGNN hidden_dim
        if input_dim is not None and input_dim != config.hidden_dim:
            self.input_projection = nn.Linear(input_dim, config.hidden_dim)
        else:
            self.input_projection = None
            
        self.layers = nn.ModuleList([
            EGNNLayer(
                node_dim=config.hidden_dim,
                edge_dim=config.edge_dim,
                coord_updates=config.coordinate_updates,
                residual=config.residual,
                layer_norm=config.layer_norm,
                dropout=config.dropout
            ) for _ in range(config.num_layers)
        ])

    def forward(
        self,
        node_features: torch.Tensor,
        coordinates: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = node_features
        x = coordinates
        
        # Project input features to hidden_dim if needed
        if self.input_projection is not None:
            h = self.input_projection(h)
        
        # Generate default edge attributes if config specifies edge_dim > 0 and none provided
        if self.config.edge_dim > 0 and edge_attr is None:
            edge_attr = torch.zeros((edge_index.size(1), self.config.edge_dim), device=h.device, dtype=h.dtype)
            
        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_attr)
            
        return h, x
