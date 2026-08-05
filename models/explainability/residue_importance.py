import torch
from typing import List, Dict, Any

def map_atoms_to_residues(atom_scores: torch.Tensor, protein_batch: Any) -> torch.Tensor:
    """
    Utility function to map atom scores or node attributions.
    """
    return atom_scores

def extract_top_residues(residue_scores: torch.Tensor, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Extracts top important residues based on attribution scores.
    """
    scores = residue_scores.tolist()
    sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
    
    top_residues = []
    for idx in sorted_indices[:top_k]:
        top_residues.append({
            "residue_index": idx,
            "residue_name": f"RES_{idx}",
            "score": scores[idx]
        })
    return top_residues
