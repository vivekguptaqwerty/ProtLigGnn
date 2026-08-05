import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List

def compute_faithfulness_curves(
    model: nn.Module,
    ligand_batch: Any,
    protein_batch: Any,
    atom_scores: torch.Tensor,
    residue_scores: torch.Tensor,
    steps: int = 10
) -> Dict[str, Any]:
    """
    Computes Deletion Curve (Comprehensiveness) and Insertion Curve (Sufficiency) for explanation faithfulness.
    Returns the curves and AUC values.
    """
    model.eval()
    
    with torch.no_grad():
        base_pred = model(ligand_batch, protein_batch).item()
        
    L = ligand_batch.x.size(0)
    P = protein_batch.x.size(0)
    
    # Sort indices by score descending
    sorted_atoms = torch.argsort(atom_scores, descending=True).tolist()
    sorted_residues = torch.argsort(residue_scores, descending=True).tolist()
    
    # 1. Deletion Curve (Comprehensiveness)
    # Sequentially zero out the most important atoms and residues
    deletion_vals = [base_pred]
    
    # We step through steps percentages (e.g. 10%, 20%, ..., 100%)
    for step in range(1, steps + 1):
        pct = step / steps
        num_atoms_to_del = int(pct * L)
        num_residues_to_del = int(pct * P)
        
        ligand_occ = ligand_batch.clone()
        ligand_occ.x = ligand_batch.x.clone()
        for idx in sorted_atoms[:num_atoms_to_del]:
            ligand_occ.x[idx] = 0
            
        protein_occ = protein_batch.clone()
        protein_occ.x = protein_batch.x.clone()
        for idx in sorted_residues[:num_residues_to_del]:
            protein_occ.x[idx] = 0
            
        with torch.no_grad():
            pred = model(ligand_occ, protein_occ).item()
        deletion_vals.append(pred)

    # 2. Insertion Curve (Sufficiency)
    # Start with empty (all masked) GNN inputs and insert the most important atoms/residues
    insertion_vals = []
    
    # Mask all baseline
    ligand_base = ligand_batch.clone()
    ligand_base.x = torch.zeros_like(ligand_batch.x)
    protein_base = protein_batch.clone()
    protein_base.x = torch.zeros_like(protein_batch.x)
    
    with torch.no_grad():
        start_pred = model(ligand_base, protein_base).item()
    insertion_vals.append(start_pred)
    
    for step in range(1, steps + 1):
        pct = step / steps
        num_atoms_to_ins = int(pct * L)
        num_residues_to_ins = int(pct * P)
        
        ligand_occ = ligand_batch.clone()
        ligand_occ.x = torch.zeros_like(ligand_batch.x)
        for idx in sorted_atoms[:num_atoms_to_ins]:
            ligand_occ.x[idx] = ligand_batch.x[idx]
            
        protein_occ = protein_batch.clone()
        protein_occ.x = torch.zeros_like(protein_batch.x)
        for idx in sorted_residues[:num_residues_to_ins]:
            protein_occ.x[idx] = protein_batch.x[idx]
            
        with torch.no_grad():
            pred = model(ligand_occ, protein_occ).item()
        insertion_vals.append(pred)

    # Calculate Areas Under the Curves (AUC) using trapezoidal rule
    deletion_auc = float(np.trapz(deletion_vals, dx=1.0/steps))
    insertion_auc = float(np.trapz(insertion_vals, dx=1.0/steps))
    
    # Comprehensiveness = BasePred - DelPred_final
    comprehensiveness = base_pred - deletion_vals[-1]
    # Sufficiency = InsPred_final - BasePred
    sufficiency = insertion_vals[-1] - base_pred
    
    return {
        "deletion_curve": deletion_vals,
        "insertion_curve": insertion_vals,
        "deletion_auc": deletion_auc,
        "insertion_auc": insertion_auc,
        "comprehensiveness": comprehensiveness,
        "sufficiency": sufficiency
    }

def compute_infidelity_and_sensitivity(
    model: nn.Module,
    ligand_batch: Any,
    protein_batch: Any,
    atom_scores: torch.Tensor,
    residue_scores: torch.Tensor,
    n_perturbations: int = 15
) -> Dict[str, float]:
    """
    Computes explanation infidelity (robustness to noise) and sensitivity (maximum variation).
    """
    # 1. Infidelity: expected squared difference between prediction change and explanation projection
    infidelity = 0.0
    sensitivity = 0.0
    
    L = ligand_batch.x.size(0)
    P = protein_batch.x.size(0)
    
    with torch.no_grad():
        base_pred = model(ligand_batch, protein_batch).item()
        
    for _ in range(n_perturbations):
        # Apply minor Gaussian noise perturbation to ligand and protein coordinates
        l_noise = torch.randn_like(ligand_batch.pos) * 0.05
        p_noise = torch.randn_like(protein_batch.pos) * 0.05
        
        ligand_p = ligand_batch.clone()
        ligand_p.pos = ligand_batch.pos + l_noise
        protein_p = protein_batch.clone()
        protein_p.pos = protein_batch.pos + p_noise
        
        with torch.no_grad():
            pert_pred = model(ligand_p, protein_p).item()
            
        pred_diff = base_pred - pert_pred
        
        # Project noise onto explanation
        # Explanation projection = sum(noise * attributions)
        # Flatten and align shape
        l_proj = (l_noise.norm(dim=-1) * atom_scores).sum().item()
        p_proj = (p_noise.norm(dim=-1) * residue_scores).sum().item()
        proj_sum = l_proj + p_proj
        
        infidelity += (pred_diff - proj_sum) ** 2
        sensitivity = max(sensitivity, abs(pred_diff))
        
    infidelity = infidelity / n_perturbations
    
    return {
        "infidelity": float(infidelity),
        "sensitivity": float(sensitivity)
    }
