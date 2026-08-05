import torch
import torch.nn as nn
from models.soft_routing.config import RoutingConfig
from models.soft_routing.router import TaskRouter

def get_activation(act_str: str) -> nn.Module:
    if act_str.lower() == "gelu":
        return nn.GELU()
    return nn.ReLU()

def get_normalization(norm_str: str, dim: int) -> nn.Module:
    if norm_str.lower() == "layer":
        return nn.LayerNorm(dim)
    return nn.Identity()

class LinearRouter(TaskRouter):
    """Simple linear projection router."""
    
    def __init__(self, config: RoutingConfig) -> None:
        super().__init__(config)
        self.proj = nn.Linear(config.latent_dimension, config.routing_dimension)
        
    def forward(self, z: torch.Tensor) -> tuple:
        return self.proj(z), None

class ResidualRouter(TaskRouter):
    """Direct projection with residual add."""
    
    def __init__(self, config: RoutingConfig) -> None:
        super().__init__(config)
        self.proj = nn.Linear(config.latent_dimension, config.routing_dimension)
        
    def forward(self, z: torch.Tensor) -> tuple:
        return z + self.proj(z), None

class GatedRouter(TaskRouter):
    """Gated router computing dynamic gating values."""
    
    def __init__(self, config: RoutingConfig) -> None:
        super().__init__(config)
        
        # Scaling capacity
        capacity = config.router_capacity.lower()
        if capacity == "tiny":
            hidden_dim = max(8, config.routing_dimension // 4)
        elif capacity == "small":
            hidden_dim = max(16, config.routing_dimension // 2)
        else:
            hidden_dim = config.routing_dimension
            
        layers = []
        in_dim = config.latent_dimension
        
        # Build gating MLP layers
        for _ in range(config.num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(get_normalization(config.normalization, hidden_dim))
            layers.append(get_activation(config.activation))
            layers.append(nn.Dropout(config.dropout))
            in_dim = hidden_dim
            
        layers.append(nn.Linear(in_dim, config.routing_dimension))
        self.gate_mlp = nn.Sequential(*layers)
        
        self.proj = nn.Linear(config.latent_dimension, config.routing_dimension)
        
    def forward(self, z: torch.Tensor) -> tuple:
        # compute gate
        gate_logits = self.gate_mlp(z)
        gate = torch.sigmoid(gate_logits)
        
        # residual output
        residual = self.proj(z)
        routed_z = gate * z + (1.0 - gate) * residual
        
        return routed_z, gate
