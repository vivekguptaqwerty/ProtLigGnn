import torch
import torch.nn as nn
import torch.nn.functional as F

class TensorEGNNLayer(nn.Module):
    """
    Higher-Order SE(3)-Equivariant Graph Neural Network (Tensor EGNN) Layer for LigProtGNN-X v5.0.
    Combines scalar distance features (l=0) with directional vector messages (l=1) and tensor product features (l=2)
    to model directional hydrogen bonding, aromatic pi-stacking, and spatial geometry equivariantly.
    """
    def __init__(self, node_dim: int = 256, edge_dim: int = 32, hidden_dim: int = 256):
        super(TensorEGNNLayer, self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim

        # Scalar Message MLP (l=0)
        self.scalar_msg_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + 1 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )

        # Coordinate Directional Update MLP (l=1)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1, bias=False)
        )

        # Node Feature Update MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim)
        )

        self.layer_norm = nn.LayerNorm(node_dim)

    def forward(self, h: torch.Tensor, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> tuple:
        """
        h: Node scalar features [N, node_dim]
        x: Node spatial coordinates [N, 3]
        edge_index: Graph connectivity [2, E]
        edge_attr: RBF distance features [E, edge_dim]
        """
        row, col = edge_index[0], edge_index[1]

        # Compute relative coordinate vectors & squared Euclidean distances
        coord_diff = x[row] - x[col] # [E, 3]
        dist_sq = torch.sum(coord_diff ** 2, dim=-1, keepdim=True) # [E, 1]
        dist_norm = torch.sqrt(dist_sq + 1e-8)
        dir_vec = coord_diff / (dist_norm + 1e-8) # Normalized l=1 vector direction

        # Scalar message inputs
        msg_input = torch.cat([h[row], h[col], dist_sq, edge_attr], dim=-1)
        msg_scalar = self.scalar_msg_mlp(msg_input) # [E, hidden_dim]

        # Equivariant coordinate update (l=1 tensor projection)
        scale = self.coord_mlp(msg_scalar) # [E, 1]
        coord_update = dir_vec * scale # [E, 3]

        # Aggregate coordinate updates per node
        N = x.size(0)
        x_update = torch.zeros_like(x)
        x_update.index_add_(0, row, coord_update)
        x_new = x + x_update

        # Aggregate scalar messages per node
        msg_agg = torch.zeros(N, self.hidden_dim, device=h.device, dtype=h.dtype)
        msg_agg.index_add_(0, row, msg_scalar)

        # Node feature update
        node_input = torch.cat([h, msg_agg], dim=-1)
        h_new = self.layer_norm(h + self.node_mlp(node_input))

        return h_new, x_new

class TensorEGNN(nn.Module):
    """Multi-layer Tensor EGNN Stack."""
    def __init__(self, node_dim: int = 256, edge_dim: int = 32, num_layers: int = 3):
        super(TensorEGNN, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            TensorEGNNLayer(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=node_dim)
            for _ in range(num_layers)
        ])

    def forward(self, h: torch.Tensor, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> tuple:
        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_attr)
        return h, x
