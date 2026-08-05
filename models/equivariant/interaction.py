import torch
from typing import Tuple, Dict, Any

def construct_complex_graph(
    ligand_h: torch.Tensor,
    ligand_x: torch.Tensor,
    ligand_edge_index: torch.Tensor,
    protein_h: torch.Tensor,
    protein_x: torch.Tensor,
    protein_edge_index: torch.Tensor,
    contact_threshold: float = 8.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Constructs a unified, geometric complex graph by combining protein pocket residues 
    and ligand atoms, adding contact edges within the specified distance threshold.
    """
    N_L = ligand_h.size(0)
    N_P = protein_h.size(0)
    
    # 1. Combine node features and coordinates
    h_complex = torch.cat([ligand_h, protein_h], dim=0)
    x_complex = torch.cat([ligand_x, protein_x], dim=0)
    
    # 2. Combine internal GNN edge indices (shifting protein indices by N_L)
    shifted_protein_edge = protein_edge_index + N_L
    gnn_edges = torch.cat([ligand_edge_index, shifted_protein_edge], dim=1)
    
    # 3. Compute bipartite interaction edges based on contact threshold
    dist_matrix = torch.cdist(ligand_x, protein_x)  # Shape (N_L, N_P)
    row_idx, col_idx = torch.where(dist_matrix <= contact_threshold)
    
    # Map to complex node indices (ligand at 0..N_L-1, protein at N_L..N_L+N_P-1)
    ligand_indices = row_idx
    protein_indices = col_idx + N_L
    
    # Add bidirectional edges
    bipartite_edges_forward = torch.stack([ligand_indices, protein_indices], dim=0)
    bipartite_edges_backward = torch.stack([protein_indices, ligand_indices], dim=0)
    
    edge_index_complex = torch.cat([gnn_edges, bipartite_edges_forward, bipartite_edges_backward], dim=1)
    
    return h_complex, x_complex, edge_index_complex


def compute_drift_diagnostics(
    coords_in: torch.Tensor,
    coords_out: torch.Tensor
) -> Dict[str, Any]:
    """
    Computes coordinate drift, updates variance, and distance preservation statistics.
    """
    delta = coords_out - coords_in
    drift_magnitudes = torch.norm(delta, dim=-1)
    
    # Distance preservation error
    dist_in = torch.cdist(coords_in, coords_in)
    dist_out = torch.cdist(coords_out, coords_out)
    dist_preservation_error = torch.mean(torch.abs(dist_out - dist_in))
    
    return {
        "max_drift": float(torch.max(drift_magnitudes).item()),
        "mean_drift": float(torch.mean(drift_magnitudes).item()),
        "var_drift": float(torch.var(drift_magnitudes).item()),
        "max_coord_norm": float(torch.max(torch.norm(coords_out, dim=-1)).item()),
        "mean_coord_norm": float(torch.mean(torch.norm(coords_out, dim=-1)).item()),
        "distance_preservation_error": float(dist_preservation_error.item())
    }
