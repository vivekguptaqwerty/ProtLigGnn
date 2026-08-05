import torch

def compute_pairwise_distances(ligand_pos: torch.Tensor, protein_pos: torch.Tensor) -> torch.Tensor:
    """Computes the pairwise Euclidean distance matrix between ligand and protein nodes.
    
    Args:
        ligand_pos: Tensor of shape [L, 3] representing ligand atom positions.
        protein_pos: Tensor of shape [P, 3] representing protein residue positions.
        
    Returns:
        Pairwise distance tensor of shape [L, P].
    """
    # ligand_pos is [L, 3], protein_pos is [P, 3]
    # Expand to [L, 1, 3] and [1, P, 3] to broadcast subtraction
    diff = ligand_pos.unsqueeze(1) - protein_pos.unsqueeze(0)  # [L, P, 3]
    return torch.norm(diff, p=2, dim=-1)  # [L, P]
