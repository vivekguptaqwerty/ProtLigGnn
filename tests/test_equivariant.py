import pytest
import math
import torch
from torch_geometric.data import Data, Batch
from models.equivariant.config import EquivariantConfig
from models.equivariant.egnn import EGNN, EGNNLayer
from models.equivariant.model import EquivariantMultimodalGNN
from models.foundation.hybrid.model import HybridMultimodalGNN
from models.foundation.hybrid.config import HybridFoundationConfig
from models.foundation.protein.model import HybridProteinGNN
from models.geometry_attention import ProtLigGNNGeometryAttention, AttentionBiasConfig

def get_random_rotation_matrix() -> torch.Tensor:
    """Generates a random 3D rotation matrix in SO(3)."""
    q, r = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def test_equivariant_config():
    cfg = EquivariantConfig(equivariant_model="egnn", num_layers=4, hidden_dim=256)
    assert cfg.equivariant_model == "egnn"
    assert cfg.num_layers == 4
    
    with pytest.raises(ValueError):
        EquivariantConfig(equivariant_model="invalid")


def test_egnn_layer_forward():
    h = torch.randn(10, 256)
    x = torch.randn(10, 3)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    
    layer = EGNNLayer(node_dim=256, edge_dim=0, coord_updates=True, residual=True)
    h_out, x_out = layer(h, x, edge_index)
    
    assert h_out.shape == (10, 256)
    assert x_out.shape == (10, 3)


@torch.no_grad()
def test_rotation_equivariance():
    """
    Verifies that EGNN layer outputs are SE(3)-equivariant under random SO(3) rotation.
    Uses eval() mode to disable stochastic dropout masks between the two forward passes.
    """
    torch.manual_seed(42)
    h = torch.randn(8, 64)
    x = torch.randn(8, 3)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 0, 3, 2, 5, 4]], dtype=torch.long)
    
    layer = EGNNLayer(node_dim=64, edge_dim=0, coord_updates=True, residual=True)
    layer.eval()  # Disable dropout for deterministic equivariance check
    
    # Baseline forward
    h_base, x_base = layer(h.clone(), x.clone(), edge_index)
    
    # Rotate coordinates
    R = get_random_rotation_matrix()
    x_rotated = x @ R  # Right-multiply: each row vector rotated
    h_rot, x_rot = layer(h.clone(), x_rotated.clone(), edge_index)
    
    # Features must be invariant (only depend on dist_sq which is rotation-invariant)
    assert torch.allclose(h_base, h_rot, atol=1e-5), \
        f"Feature invariance failed. Max diff: {(h_base - h_rot).abs().max():.2e}"
    
    # Coordinates must be equivariant: rotate(output) ≈ output_of_rotated
    assert torch.allclose(x_base @ R, x_rot, atol=1e-5), \
        f"Coordinate equivariance failed. Max diff: {(x_base @ R - x_rot).abs().max():.2e}"


@torch.no_grad()
def test_translation_equivariance():
    """
    Verifies that EGNN layer outputs are SE(3)-equivariant under translation.
    Uses eval() mode to disable stochastic dropout masks between the two forward passes.
    """
    torch.manual_seed(42)
    h = torch.randn(8, 64)
    x = torch.randn(8, 3)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 0, 3, 2, 5, 4]], dtype=torch.long)
    
    layer = EGNNLayer(node_dim=64, edge_dim=0, coord_updates=True, residual=True)
    layer.eval()
    
    h_base, x_base = layer(h.clone(), x.clone(), edge_index)
    
    T = torch.tensor([[5.0, -3.0, 7.0]])
    x_translated = x + T
    h_trans, x_trans = layer(h.clone(), x_translated.clone(), edge_index)
    
    # Features invariant under translation
    assert torch.allclose(h_base, h_trans, atol=1e-5), \
        f"Translation feature invariance failed. Max diff: {(h_base - h_trans).abs().max():.2e}"
    
    # Coordinates equivariant: output + T ≈ output_of_translated
    assert torch.allclose(x_base + T, x_trans, atol=1e-5), \
        f"Translation coordinate equivariance failed. Max diff: {(x_base + T - x_trans).abs().max():.2e}"


@torch.no_grad()
def test_reflection_diagnostic():
    """
    Diagnostic test for reflection behavior.
    EGNN messages use squared distances, which are reflection-invariant.
    This test documents the observed behavior under O(3) reflections.
    Reflections are NOT promotion criteria for SE(3) implementations.
    """
    torch.manual_seed(42)
    h = torch.randn(6, 64)
    x = torch.randn(6, 3)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    
    layer = EGNNLayer(node_dim=64, edge_dim=0, coord_updates=True)
    layer.eval()
    
    h_base, x_base = layer(h.clone(), x.clone(), edge_index)
    
    # Reflect across X-axis
    x_reflected = x.clone()
    x_reflected[:, 0] = -x_reflected[:, 0]
    h_ref, x_ref = layer(h.clone(), x_reflected.clone(), edge_index)
    
    # Features should be invariant since messages only use dist_sq
    feat_diff = (h_base - h_ref).abs().max().item()
    print(f"[Diagnostic] Reflection feature max diff: {feat_diff:.2e}")
    assert feat_diff < 1e-4, f"Reflection feature diff unexpectedly large: {feat_diff:.2e}"


def test_numerical_stability():
    """Tests EGNN with degenerate coordinates (zero distance)."""
    h = torch.randn(4, 64)
    x = torch.zeros(4, 3)  # All nodes at origin
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    
    layer = EGNNLayer(node_dim=64, edge_dim=0, coord_updates=True)
    h_out, x_out = layer(h, x, edge_index)
    
    assert not torch.isnan(h_out).any(), "NaN in features"
    assert not torch.isinf(x_out).any(), "Inf in coordinates"


def test_gradient_flow():
    """Tests that gradients flow through the EGNN stack."""
    config = EquivariantConfig(num_layers=2, hidden_dim=64, edge_dim=0)
    egnn = EGNN(config)
    
    h = torch.randn(6, 64, requires_grad=True)
    x = torch.randn(6, 3, requires_grad=True)
    edge_index = torch.tensor([[0, 1, 2], [1, 0, 2]], dtype=torch.long)
    
    h_out, x_out = egnn(h, x, edge_index)
    loss = h_out.sum() + x_out.sum()
    loss.backward()
    
    assert h.grad is not None, "No gradient on input features"
    assert x.grad is not None, "No gradient on input coordinates"
    for name, param in egnn.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient on {name}"


def test_egnn_with_input_projection():
    """Tests EGNN when input_dim != hidden_dim, requiring projection."""
    config = EquivariantConfig(num_layers=2, hidden_dim=128, edge_dim=0)
    egnn = EGNN(config, input_dim=512)
    
    assert egnn.input_projection is not None
    
    h = torch.randn(8, 512)
    x = torch.randn(8, 3)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    
    h_out, x_out = egnn(h, x, edge_index)
    
    assert h_out.shape == (8, 128), f"Expected (8, 128), got {h_out.shape}"
    assert x_out.shape == (8, 3)


def test_equivariant_multimodal_gnn():
    """Full integration test: EquivariantMultimodalGNN forward pass + gradient flow."""
    bias_config = AttentionBiasConfig(
        bias_type="rbf", num_rbf=16, cutoff_distance=12.0,
        learnable_scale=True, initial_alpha=1.0, normalize_bias=True, num_heads=4
    )
    base_gnn = ProtLigGNNGeometryAttention(
        ligand_dim=5, protein_dim=5, hidden_dim=256,
        no_crossgraph=False, config=bias_config
    )
    
    from models.foundation.protein.config import ProteinFoundationConfig
    prot_config = ProteinFoundationConfig(
        protein_model="esm2", latent_dimension=256,
        projection_dimension=256, freeze_backbone=True
    )
    base_hybrid_prot = HybridProteinGNN(base_gnn=base_gnn, config=prot_config)
    
    hybrid_config = HybridFoundationConfig(
        protein_model="esm2", ligand_model="chemberta",
        fusion_strategy="cross_attention", projection_dimension=256
    )
    base_hybrid = HybridMultimodalGNN(config=hybrid_config, base_hybrid=base_hybrid_prot)
    
    equiv_config = EquivariantConfig(
        equivariant_model="egnn", num_layers=2, hidden_dim=256,
        edge_dim=32, coordinate_updates=True, residual=True, layer_norm=True
    )
    
    model = EquivariantMultimodalGNN(config=equiv_config, base_hybrid=base_hybrid)
    
    # Mock data batches
    p_graph = Data(
        x=torch.randn(10, 5),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        pos=torch.randn(10, 3)
    )
    p_graph.foundation_emb = torch.randn(10, 320)
    p_batch = Batch.from_data_list([p_graph])
    
    l_graph = Data(
        x=torch.randn(6, 5),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        pos=torch.randn(6, 3)
    )
    l_graph.foundation_emb = torch.randn(6, 384)
    l_batch = Batch.from_data_list([l_graph])
    
    # Forward pass
    pred = model(l_batch, p_batch)
    assert pred.shape == (1,), f"Expected shape (1,), got {pred.shape}"
    
    # Gradient flow check
    loss = pred.sum()
    loss.backward()
    
    assert next(model.egnn.parameters()).grad is not None, "No gradient on EGNN params"
    assert next(model.base_model.projection_ligand.parameters()).grad is not None, "No gradient on ligand projection"
    
    # Verify coordinate drift diagnostics were captured
    assert model.last_drift_diagnostics is not None
    assert "max_drift" in model.last_drift_diagnostics
    assert "distance_preservation_error" in model.last_drift_diagnostics
