import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from models.soft_routing.config import RoutingConfig

class TaskRouter(nn.Module, ABC):
    """Abstract interface for all task routers."""
    
    def __init__(self, config: RoutingConfig) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, z: torch.tensor) -> tuple:
        """
        Routes the shared interaction embedding.
        
        Args:
            z: [L, P, D] or [L * P, D] shared representation
            
        Returns:
            Tuple of (routed_z, gate_tensor)
        """
        pass
