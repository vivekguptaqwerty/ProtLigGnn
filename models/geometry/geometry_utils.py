from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import torch

@dataclass(frozen=True)
class GeometryFeatures:
    """Immutable container for geometric features.
    
    API designed for future extensibility to coordinate, direction, and angle 
    learning without breaking changes to down-stream layers.
    """
    distances: torch.Tensor
    relative_vectors: Optional[torch.Tensor] = None
    directions: Optional[torch.Tensor] = None
    edge_mask: Optional[torch.Tensor] = None
    edge_index: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def compute_geometric_features(pos: torch.Tensor, edge_index: torch.Tensor) -> GeometryFeatures:
    """Computes explicit geometric features for a graph on the fly.
    
    Calculates Euclidean distances for each graph edge from node coordinates.
    
    Args:
        pos: Node coordinates of shape [num_nodes, 3].
        edge_index: Directed edge pairs of shape [2, num_edges].
        
    Returns:
        A GeometryFeatures container with computed distance values.
    """
    if edge_index.numel() == 0:
        # Graceful handling of graphs with no edges
        distances = torch.empty(0, dtype=torch.float32, device=pos.device)
        return GeometryFeatures(distances=distances, edge_index=edge_index)
        
    u, v = edge_index[0], edge_index[1]
    
    # Compute Euclidean distance: d_ij = ||p_i - p_j||_2
    diffs = pos[u] - pos[v]
    distances = torch.norm(diffs, p=2, dim=-1)
    
    return GeometryFeatures(
        distances=distances,
        edge_index=edge_index
    )
