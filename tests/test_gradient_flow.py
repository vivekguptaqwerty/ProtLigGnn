import torch
import pytest
from torch_geometric.data import Data, Batch
from models.physics_guided import ProtLigGNNPhysicsGuided, PhysicsGuidedConfig

def create_mock_batch():
    x_l = torch.randn(3, 78)
    edge_l = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    pos_l = torch.randn(3, 3)
    ligand = Data(x=x_l, edge_index=edge_l, pos=pos_l)
    
    x_p = torch.randn(4, 21)
    edge_p = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    pos_p = torch.randn(4, 3)
    protein = Data(x=x_p, edge_index=edge_p, pos=pos_p)
    
    return Batch.from_data_list([ligand]), Batch.from_data_list([protein])

def test_gradient_flow_multitask():
    model = ProtLigGNNPhysicsGuided(
        ligand_dim=78,
        protein_dim=21,
        hidden_dim=32,
        config=PhysicsGuidedConfig(contact_hidden_dim=32, loss_weight_strategy="learnable")
    )
    
    ligand_batch, protein_batch = create_mock_batch()
    preds, contact_logits, contact_targets, contexts = model(ligand_batch, protein_batch)
    
    # 1. Affinity Loss
    aff_loss = torch.mean((preds - torch.tensor([5.0]))**2)
    # 2. Contact Loss
    contact_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        contact_logits[0], contact_targets[0]
    )
    
    # Total loss
    total_loss, _ = model.multitask_loss(aff_loss, contact_loss)
    
    # Backprop
    total_loss.backward()
    
    # Ensure gradients propagate to encoders and heads
    assert any(p.grad is not None for p in model.base_model.ligand_encoder.conv1.parameters())
    assert model.base_model.cross_attention.ligand_to_protein_attn.in_proj_weight.grad is not None
    assert model.contact_head.mlp[0].weight.grad is not None
    assert model.base_model.regressor[0].weight.grad is not None
    assert model.multitask_loss.log_lambda.grad is not None
