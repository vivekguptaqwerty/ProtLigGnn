import torch
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass(frozen=True)
class RoutingContext:
    shared_embedding: torch.Tensor
    routed_affinity_embedding: torch.Tensor
    routed_contact_embedding: torch.Tensor
    affinity_gates: Optional[torch.Tensor] = None
    contact_gates: Optional[torch.Tensor] = None
    routing_statistics: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    coordinates: Optional[Dict[str, torch.Tensor]] = None
    predicted_interaction_embeddings: Optional[torch.Tensor] = None
