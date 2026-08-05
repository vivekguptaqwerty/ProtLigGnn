import torch
import torch.nn as nn
from typing import Tuple, Any

def compute_raw_gradients(
    model: nn.Module,
    ligand_batch: Any,
    protein_batch: Any
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes gradients of the model prediction with respect to:
    1. Ligand node representations
    2. Protein node representations
    3. Spatial coordinates (positions)
    """
    model.eval()
    
    # 1. Enable grad tracking on positions
    ligand_pos = ligand_batch.pos.clone().detach().requires_grad_(True)
    protein_pos = protein_batch.pos.clone().detach().requires_grad_(True)
    
    # Clone and track features
    ligand_batch_t = ligand_batch.clone()
    protein_batch_t = protein_batch.clone()
    ligand_batch_t.pos = ligand_pos
    protein_batch_t.pos = protein_pos
    
    # Track representation gradients using a forward activation hook or intermediate hook
    # Since we want node representation gradients, we can register hooks or run the model sub-modules manually.
    # A clean way is to run the model and compute gradients w.r.t the GNN encoder outputs!
    if hasattr(model, "base_model"):
        gnn_model = model.base_model
        if hasattr(gnn_model, "base_model"):
            gnn_model = gnn_model.base_model
    else:
        gnn_model = model

    # Forward encoders
    ligand_x = gnn_model.ligand_encoder(ligand_batch_t.x, ligand_batch_t.edge_index).clone().detach().requires_grad_(True)
    protein_x = gnn_model.protein_encoder(protein_batch_t.x, protein_batch_t.edge_index).clone().detach().requires_grad_(True)
    
    # Reconstruct prediction from encoders
    # Perform cross-graph interaction
    ligand_inter, protein_inter = gnn_model.cross_graph_interaction(
        ligand_x, protein_x, ligand_pos, protein_pos
    )
    
    # Pooling
    from torch_geometric.nn import global_mean_pool
    ligand_pool = global_mean_pool(ligand_inter, ligand_batch_t.batch)
    protein_pool = global_mean_pool(protein_inter, protein_batch_t.batch)
    
    joint = gnn_model.dropout(torch.cat([ligand_pool, protein_pool], dim=-1))
    prediction = gnn_model.regressor(joint).squeeze(-1)
    
    # Compute backprop of predictions
    grad_outputs = torch.ones_like(prediction)
    grads = torch.autograd.grad(
        outputs=prediction,
        inputs=[ligand_x, protein_x, ligand_pos],
        grad_outputs=grad_outputs,
        create_graph=False,
        retain_graph=False,
        allow_unused=True
    )
    
    grad_ligand_x = grads[0] if grads[0] is not None else torch.zeros_like(ligand_x)
    grad_protein_x = grads[1] if grads[1] is not None else torch.zeros_like(protein_x)
    grad_ligand_pos = grads[2] if grads[2] is not None else torch.zeros_like(ligand_pos)
    
    return grad_ligand_x, grad_protein_x, grad_ligand_pos
