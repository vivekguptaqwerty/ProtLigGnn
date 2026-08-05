import torch

def generate_contact_targets(pos_lig: torch.Tensor, pos_prot: torch.Tensor, 
                             strategy: str = "binary", threshold: float = 4.0,
                             gamma: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """Generates pairwise interaction/contact target matrix between ligand atoms and protein residues.
    
    Supported strategies:
    - 'binary': Binary contact map where distance < threshold (Default)
    - 'gaussian': Continuous Gaussian labels exp(-gamma * d^2)
    - 'distance_decay': Continuous 1 / (1 + d^2)
    - 'inverse_distance': Continuous 1 / (d + eps)
    """
    # pos_lig: [L, 3], pos_prot: [P, 3] -> dists: [L, P]
    dists = torch.cdist(pos_lig, pos_prot)
    
    if strategy == "binary":
        return (dists < threshold).float()
    elif strategy == "gaussian":
        return torch.exp(-gamma * (dists ** 2))
    elif strategy == "distance_decay":
        return 1.0 / (1.0 + (dists ** 2))
    elif strategy == "inverse_distance":
        return 1.0 / (dists + eps)
    else:
        raise ValueError(f"Unknown target generation strategy: {strategy}")
