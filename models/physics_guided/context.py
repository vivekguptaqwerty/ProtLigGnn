import torch
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass(frozen=True)
class PhysicsContext:
    """Immutable communication object containing geometric and latent context for a batch."""
    coordinates: Dict[str, torch.Tensor]
    node_embeddings: Dict[str, torch.Tensor]
    batch_indices: Dict[str, torch.Tensor]
    masks: Dict[str, torch.Tensor]
    predicted_interaction_embeddings: torch.Tensor
    metadata: Dict[str, Any] = field(default_factory=dict)
