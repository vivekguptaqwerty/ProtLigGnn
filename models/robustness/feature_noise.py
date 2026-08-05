import torch
from typing import Any

def inject_feature_noise(batch: Any, fraction: float, max_token_val: int = 20) -> Any:
    """
    Randomly perturbs/swaps a fraction of discrete node features (long tokens) in the Batch object.
    If fraction is 0.0, it is a strict identity operation.
    """
    perturbed_batch = batch.clone()
    if fraction > 0.0:
        x = batch.x.clone()
        mask = torch.rand(x.shape[0]) < fraction
        # Replace selected indices with random tokens within valid range
        random_tokens = torch.randint(0, max_token_val, (mask.sum().item(),), dtype=x.dtype, device=x.device)
        
        # If multidimensional token indices, select first column or apply to all
        if len(x.shape) > 1:
            x[mask, 0] = random_tokens
        else:
            x[mask] = random_tokens
            
        perturbed_batch.x = x
    return perturbed_batch
