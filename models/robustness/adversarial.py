import torch
import torch.nn as nn
from typing import Any

def generate_adversarial_coordinates(
    model: nn.Module,
    ligand_batch: Any,
    protein_batch: Any,
    epsilon: float = 0.05,
    steps: int = 5
) -> Any:
    """
    Generates adversarial coordinate perturbations using PGD step to maximize model error/loss.
    """
    model.eval()
    
    # Enable gradient tracking on ligand coordinates
    perturbed_ligand = ligand_batch.clone()
    ligand_pos = ligand_batch.pos.clone().detach().requires_grad_(True)
    perturbed_ligand.pos = ligand_pos
    
    with torch.enable_grad():
        for _ in range(steps):
            # Run prediction
            pred = model(perturbed_ligand, protein_batch)
            loss = pred.sum()  # Maximize the predictions sum or direct error
            
            grads = torch.autograd.grad(loss, ligand_pos, allow_unused=True)
            if grads[0] is not None:
                grad_sign = grads[0].sign()
                # Gradient ascent step: pos = pos + epsilon * grad_sign
                ligand_pos = ligand_pos.detach() + epsilon * grad_sign
                ligand_pos.requires_grad_(True)
                perturbed_ligand.pos = ligand_pos
                
    perturbed_ligand.pos = ligand_pos.detach()
    return perturbed_ligand
