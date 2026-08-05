import torch
from typing import Any

def apply_node_masking(batch: Any, fraction: float) -> Any:
    """
    Randomly masks a fraction of nodes (atoms or residues) by setting their features to 0.
    Zero-fraction masking is an identity operation.
    """
    masked_batch = batch.clone()
    if fraction > 0.0:
        x = batch.x.clone()
        mask = torch.rand(x.shape[0]) < fraction
        
        # Zero out the masked elements
        if len(x.shape) > 1:
            x[mask] = 0
        else:
            x[mask] = 0
            
        masked_batch.x = x
    return masked_batch
