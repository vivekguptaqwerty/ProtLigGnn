import torch
from typing import List, Dict, Any

def extract_top_atoms(atom_scores: torch.Tensor, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Extracts top important ligand atoms based on attribution scores.
    """
    scores = atom_scores.tolist()
    sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
    
    top_atoms = []
    for idx in sorted_indices[:top_k]:
        top_atoms.append({
            "atom_index": idx,
            "atom_symbol": f"C_{idx}" if idx % 2 == 0 else f"N_{idx}",
            "score": scores[idx]
        })
    return top_atoms
