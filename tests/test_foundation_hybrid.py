import pytest
import torch
from torch_geometric.data import Data, Batch
from models.foundation.hybrid.config import HybridFoundationConfig
from models.foundation.hybrid.fusion import (
    ConcatenationFusion,
    WeightedSumFusion,
    GatedFusion,
    ResidualFusion
)
from models.foundation.hybrid.interaction import BidirectionalCrossAttention
from models.foundation.hybrid.model import HybridMultimodalGNN, HybridFoundationOnly
from models.foundation.protein.model import HybridProteinGNN
from models.geometry_attention import ProtLigGNNGeometryAttention, AttentionBiasConfig

def test_hybrid_config():
    cfg = HybridFoundationConfig(
        protein_model="esm2",
        ligand_model="chemberta",
        fusion_strategy="cross_attention",
        projection_dimension=256
    )
    assert cfg.protein_model == "esm2"
    assert cfg.ligand_model == "chemberta"
    assert cfg.fusion_strategy == "cross_attention"
    assert cfg.projection_dimension == 256
    
    with pytest.raises(ValueError):
        HybridFoundationConfig(protein_model="invalid_model")

def test_fusion_modules():
    p_rep = torch.randn(10, 256)
    l_rep = torch.randn(10, 256)
    
    # 1. Concatenation
    concat = ConcatenationFusion(dim=256)
    out_concat = concat(p_rep, l_rep)
    assert out_concat.shape == (10, 256)
    
    # 2. Weighted Sum
    w_sum = WeightedSumFusion(dim=256)
    out_w_sum = w_sum(p_rep, l_rep)
    assert out_w_sum.shape == (10, 256)
    assert w_sum.alpha.requires_grad
    
    # 3. Gated
    gated = GatedFusion(dim=256)
    out_gated = gated(p_rep, l_rep)
    assert out_gated.shape == (10, 256)
    
    # 4. Residual
    residual = ResidualFusion(dim=256)
    out_res = residual(p_rep, l_rep)
    assert out_res.shape == (10, 256)

def test_cross_attention_shapes():
    p_rep = torch.randn(15, 256)
    l_rep = torch.randn(8, 256)
    
    attn = BidirectionalCrossAttention(dim=256, num_heads=4)
    out_p, out_l = attn(p_rep, l_rep)
    
    assert out_p.shape == (15, 256)
    assert out_l.shape == (8, 256)
    assert attn.last_attn_p2l.shape == (1, 4, 15, 8)
    assert attn.last_attn_l2p.shape == (1, 4, 8, 15)

def test_hybrid_foundation_only():
    config = HybridFoundationConfig(
        protein_model="esm2",
        ligand_model="chemberta",
        fusion_strategy="cross_attention",
        projection_dimension=256
    )
    
    # Mock Batch objects
    protein_x = torch.randn(15, 5)
    protein_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    protein_graph = Data(x=protein_x, edge_index=protein_edge_index)
    protein_graph.foundation_emb = torch.randn(15, 320)
    protein_batch = Batch.from_data_list([protein_graph])
    
    ligand_x = torch.randn(8, 5)
    ligand_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    ligand_graph = Data(x=ligand_x, edge_index=ligand_edge_index)
    ligand_graph.foundation_emb = torch.randn(8, 384)
    ligand_batch = Batch.from_data_list([ligand_graph])
    
    model = HybridFoundationOnly(config=config)
    output = model(ligand_batch, protein_batch)
    
    assert output.shape == (1,)
    
    # Gradient flow check
    loss = output.sum()
    loss.backward()
    assert next(model.projection_protein.parameters()).grad is not None
    assert next(model.projection_ligand.parameters()).grad is not None

def test_hybrid_multimodal_gnn():
    bias_config = AttentionBiasConfig(
        bias_type="rbf",
        num_rbf=16,
        cutoff_distance=12.0,
        learnable_scale=True,
        initial_alpha=1.0,
        normalize_bias=True,
        num_heads=4
    )
    base_gnn = ProtLigGNNGeometryAttention(
        ligand_dim=5,
        protein_dim=5,
        hidden_dim=256,
        no_crossgraph=False,
        config=bias_config
    )
    
    from models.foundation.protein.config import ProteinFoundationConfig
    prot_config = ProteinFoundationConfig(
        protein_model="esm2",
        latent_dimension=256,
        projection_dimension=256,
        freeze_backbone=True
    )
    base_hybrid = HybridProteinGNN(base_gnn=base_gnn, config=prot_config)
    
    hybrid_config = HybridFoundationConfig(
        protein_model="esm2",
        ligand_model="chemberta",
        fusion_strategy="cross_attention",
        projection_dimension=256
    )
    
    model = HybridMultimodalGNN(config=hybrid_config, base_hybrid=base_hybrid)
    
    # Mock graph objects
    p_graph = Data(
        x=torch.randn(15, 5),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        pos=torch.randn(15, 3)
    )
    p_graph.foundation_emb = torch.randn(15, 320)
    p_batch = Batch.from_data_list([p_graph])
    
    l_graph = Data(
        x=torch.randn(8, 5),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        pos=torch.randn(8, 3)
    )
    l_graph.foundation_emb = torch.randn(8, 384)
    l_batch = Batch.from_data_list([l_graph])
    
    # Test forward pass
    output = model(l_batch, p_batch)
    assert output.shape == (1,)
    
    # Test backpropagation
    loss = output.sum()
    loss.backward()
    
    # Check that gradients flow to projection heads
    assert next(model.projection_ligand.parameters()).grad is not None
    assert next(model.base_model.projection.parameters()).grad is not None
