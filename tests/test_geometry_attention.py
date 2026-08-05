import pytest
import torch
from torch_geometric.data import Batch
from models.geometry_attention.config import AttentionBiasConfig
from models.geometry_attention.geometry_attention_model import ProtLigGNNGeometryAttention

def test_geometry_attention_model_flow() -> None:
    """Verifies forward/backward pass and shapes for ProtLigGNNGeometryAttention."""
    config = AttentionBiasConfig(
        bias_type="rbf",
        num_rbf=32,
        cutoff_distance=12.0,
        learnable_scale=True,
        initial_alpha=0.1,
        normalize_bias=True,
        num_heads=4
    )
    
    model = ProtLigGNNGeometryAttention(
        ligand_dim=78,
        protein_dim=30,
        hidden_dim=64,
        config=config
    )
    model.train()
    
    # Create mock batch data (Batch size = 2)
    # Ligand: 4 nodes, 4 edges, coords
    lig_x = torch.rand(4, 78)
    lig_edge = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    lig_pos = torch.rand(4, 3) * 10.0
    lig_batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    
    # Protein: 4 nodes, 4 edges, coords
    prot_x = torch.rand(4, 30)
    prot_edge = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    prot_pos = torch.rand(4, 3) * 10.0
    prot_batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    
    l_batch = Batch(x=lig_x, edge_index=lig_edge, pos=lig_pos, batch=lig_batch)
    p_batch = Batch(x=prot_x, edge_index=prot_edge, pos=prot_pos, batch=prot_batch)
    l_batch.num_graphs = 2
    p_batch.num_graphs = 2
    
    # Forward Pass
    out = model(l_batch, p_batch)
    assert out.shape == (2,)
    
    # Backward Pass
    loss = out.sum()
    loss.backward()
    
    # Verify gradient flow to log_alpha and the projector weights
    assert model.cross_attention.log_alpha.grad is not None
    assert torch.abs(model.cross_attention.log_alpha.grad) > 0
    assert model.cross_attention.attention_bias.projector.weight.grad is not None
    assert torch.sum(torch.abs(model.cross_attention.attention_bias.projector.weight.grad)) > 0
