import pytest
import torch
from torch_geometric.data import Data, Batch
from models.soft_routing.config import RoutingConfig
from models.soft_routing.routing_context import RoutingContext
from models.soft_routing.gated_router import GatedRouter, LinearRouter, ResidualRouter
from models.soft_routing.routing_metrics import (
    compute_linear_cka,
    compute_svcca,
    compute_gate_diagnostics,
    compute_gradient_conflict
)
from models.soft_routing.multitask_model import ProtLigGNNSoftRouting

def test_config_validation():
    cfg = RoutingConfig(router_type="gated", router_capacity="tiny")
    assert cfg.router_type == "gated"
    assert cfg.router_capacity == "tiny"
    
    with pytest.raises(ValueError):
        RoutingConfig(router_type="invalid_type")
        
    with pytest.raises(ValueError):
        RoutingConfig(routing_level="invalid_level")

def test_routers_forward():
    z = torch.randn(10, 32)
    
    # Gated
    cfg = RoutingConfig(latent_dimension=32, routing_dimension=32, router_type="gated")
    router = GatedRouter(cfg)
    routed, gate = router(z)
    assert routed.shape == (10, 32)
    assert gate.shape == (10, 32)
    assert torch.all(gate >= 0.0) and torch.all(gate <= 1.0)
    
    # Linear
    router_lin = LinearRouter(cfg)
    routed_lin, gate_lin = router_lin(z)
    assert routed_lin.shape == (10, 32)
    assert gate_lin is None
    
    # Residual
    router_res = ResidualRouter(cfg)
    routed_res, gate_res = router_res(z)
    assert routed_res.shape == (10, 32)
    assert gate_res is None

def test_router_capacity_scaling():
    z = torch.randn(5, 64)
    for cap in ["tiny", "small", "base"]:
        cfg = RoutingConfig(latent_dimension=64, routing_dimension=64, router_type="gated", router_capacity=cap)
        router = GatedRouter(cfg)
        routed, gate = router(z)
        assert routed.shape == (5, 64)
        assert gate.shape == (5, 64)

def test_cka_and_svcca():
    x = torch.randn(100, 16)
    y = torch.randn(100, 16)
    
    cka = compute_linear_cka(x, y)
    assert 0.0 <= cka <= 1.0
    
    svcca = compute_svcca(x, y)
    assert 0.0 <= svcca <= 1.0
    
    # Perfect self-alignment should be 1.0
    assert abs(compute_linear_cka(x, x) - 1.0) < 1e-5
    assert abs(compute_svcca(x, x) - 1.0) < 1e-4

def test_gradient_conflict():
    g1 = torch.tensor([1.0, 0.0, -1.0])
    g2 = torch.tensor([-2.0, 0.0, 1.0])
    
    diags = compute_gradient_conflict(g1, g2)
    assert "gradient_cosine_similarity" in diags
    assert diags["gradient_conflict_frequency"] == 1.0
    assert diags["gradient_conflict_magnitude"] > 0.0

def test_soft_routing_model_forward():
    # Build dummy inputs
    x_lig = torch.randn(10, 78)
    edge_lig = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    pos_lig = torch.randn(10, 3)
    lig_data = Data(x=x_lig, edge_index=edge_lig, pos=pos_lig)
    lig_data.num_nodes = 10
    
    x_prot = torch.randn(15, 78)
    edge_prot = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    pos_prot = torch.randn(15, 3)
    prot_data = Data(x=x_prot, edge_index=edge_prot, pos=pos_prot)
    prot_data.num_nodes = 15
    
    lig_batch = Batch.from_data_list([lig_data])
    prot_batch = Batch.from_data_list([prot_data])
    
    cfg = RoutingConfig(router_type="gated")
    model = ProtLigGNNSoftRouting(
        ligand_dim=78,
        protein_dim=78,
        hidden_dim=32,
        config=cfg
    )
    
    preds, contact_logits, contact_targets, contexts = model(lig_batch, prot_batch)
    assert preds.shape == (1,)
    assert len(contact_logits) == 1
    assert len(contact_targets) == 1
    assert contact_logits[0].shape == (10, 15)
    assert len(contexts) == 1
    
    # Test regularizer loss
    reg_loss = model.compute_regularization_loss(contexts)
    assert reg_loss.ndim == 0
    assert reg_loss.item() >= 0.0
