import torch
from typing import Any

def inject_coordinate_noise(batch: Any, sigma: float) -> Any:
    """
    Applies Gaussian coordinate noise to node position coordinates in the PyG Batch object.
    Matches configured distribution, and verifies zero-noise is an identity operation.
    """
    perturbed_batch = batch.clone()
    if sigma > 0.0:
        noise = torch.randn_like(batch.pos) * sigma
        perturbed_batch.pos = batch.pos + noise
    return perturbed_batch
