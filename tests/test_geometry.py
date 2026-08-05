import pytest
import torch
import tempfile
from pathlib import Path
from torch_geometric.data import Data, Batch

from models.geometry import ProtLigGNNGeometry, GeometryFeatures, compute_geometric_features
from models.geometry.radial_basis import GaussianRadialBasis
from models.geometry.distance_encoding import DistanceEncoder

def test_rbf_encoding() -> None:
    """Verifies Gaussian Radial Basis Function (RBF) mapping properties."""
    num_basis = 32
    rbf = GaussianRadialBasis(num_basis=num_basis, start=0.0, stop=12.0)
    
    # 1. Shape validation
    dist = torch.tensor([0.0, 3.0, 6.0, 12.0], dtype=torch.float32)
    features = rbf(dist)
    assert features.shape == (4, num_basis)
    
    # 2. Check maximum values are close to 1.0 at their respective centers
    # RBF output at exactly center should be 1.0 (exp(0) = 1.0)
    means = rbf.means
    for idx, center in enumerate(means):
        # Pass the center distance
        single_dist = torch.tensor([center], dtype=torch.float32)
        out = rbf(single_dist)
        # The coordinate at idx should be 1.0
        assert torch.isclose(out[0, idx], torch.tensor(1.0), atol=1e-5)


def test_distance_encoder() -> None:
    """Verifies DistanceEncoder projection output shapes."""
    encoder = DistanceEncoder(num_basis=32, start=0.0, stop=12.0, projection_dim=32)
    dist = torch.tensor([1.5, 4.2], dtype=torch.float32)
    out = encoder(dist)
    assert out.shape == (2, 32)


def test_geometry_features_container() -> None:
    """Verifies the GeometryFeatures container class structure and API immutability."""
    dists = torch.tensor([1.0, 2.0], dtype=torch.float32)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    
    geom = GeometryFeatures(
        distances=dists,
        edge_index=edge_index,
        metadata={"precision": "float32"}
    )
    
    assert torch.equal(geom.distances, dists)
    assert torch.equal(geom.edge_index, edge_index)
    assert geom.metadata["precision"] == "float32"
    assert geom.relative_vectors is None  # Reserved for future phases
    
    # Immutability validation (dataclass frozen)
    with pytest.raises(Exception):
        geom.distances = torch.tensor([3.0, 4.0])  # type: ignore


def test_compute_geometric_features() -> None:
    """Verifies correct distance computation from positions."""
    # 3 nodes: node0 at (0,0,0), node1 at (3,0,0), node2 at (0,4,0)
    pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [0.0, 4.0, 0.0]
    ], dtype=torch.float32)
    
    # Edges: 0->1 (dist=3.0), 0->2 (dist=4.0)
    edge_index = torch.tensor([
        [0, 0],
        [1, 2]
    ], dtype=torch.long)
    
    geom = compute_geometric_features(pos, edge_index)
    assert geom.distances.shape == (2,)
    assert torch.isclose(geom.distances[0], torch.tensor(3.0), atol=1e-5)
    assert torch.isclose(geom.distances[1], torch.tensor(4.0), atol=1e-5)
    
    # Empty edges check
    empty_edge = torch.empty((2, 0), dtype=torch.long)
    geom_empty = compute_geometric_features(pos, empty_edge)
    assert geom_empty.distances.numel() == 0


def test_geometry_model_forward_backward() -> None:
    """Verifies the forward and backward passes of ProtLigGNNGeometry."""
    model = ProtLigGNNGeometry(
        ligand_dim=78,
        protein_dim=30,
        hidden_dim=64,  # Smaller hidden size for fast testing
        num_basis=32,
        projection_dim=32
    )
    model.train()
    
    # Create dummy batch (Batch size = 2)
    # Ligand: 4 nodes, 4 edges
    lig_x = torch.rand(4, 78)
    lig_edge = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    lig_pos = torch.rand(4, 3) * 10.0
    lig_batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    
    # Protein: 4 nodes, 4 edges
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
    
    # Verify that gradients flow to RBF projector weights
    assert model.distance_encoder.projector.weight.grad is not None
    assert torch.sum(torch.abs(model.distance_encoder.projector.weight.grad)) > 0


def test_determinism() -> None:
    """Verifies that seed control yields identical predictions and features."""
    torch.manual_seed(42)
    model = ProtLigGNNGeometry(ligand_dim=78, protein_dim=30, hidden_dim=64)
    
    lig_x = torch.rand(4, 78)
    lig_edge = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    lig_pos = torch.rand(4, 3) * 10.0
    lig_batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    
    prot_x = torch.rand(4, 30)
    prot_edge = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    prot_pos = torch.rand(4, 3) * 10.0
    prot_batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    
    l_batch = Batch(x=lig_x, edge_index=lig_edge, pos=lig_pos, batch=lig_batch)
    p_batch = Batch(x=prot_x, edge_index=prot_edge, pos=prot_pos, batch=prot_batch)
    l_batch.num_graphs = 2
    p_batch.num_graphs = 2
    
    model.eval()
    with torch.no_grad():
        out1 = model(l_batch, p_batch)
        out2 = model(l_batch, p_batch)
        
    assert torch.allclose(out1, out2, atol=1e-7)


def test_checkpoint_compatibility() -> None:
    """Verifies checkpoint saving and loading functionality."""
    model = ProtLigGNNGeometry(ligand_dim=78, protein_dim=30, hidden_dim=64)
    state_dict = model.state_dict()
    
    new_model = ProtLigGNNGeometry(ligand_dim=78, protein_dim=30, hidden_dim=64)
    new_model.load_state_dict(state_dict)
    
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1, p2)


def test_gpu_compatibility() -> None:
    """Verifies execution on CUDA if a GPU is available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU is not available.")
        
    device = torch.device("cuda")
    model = ProtLigGNNGeometry(ligand_dim=78, protein_dim=30, hidden_dim=64).to(device)
    model.eval()
    
    lig_x = torch.rand(2, 78).to(device)
    lig_edge = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).to(device)
    lig_pos = torch.rand(2, 3).to(device)
    lig_batch = torch.tensor([0, 0], dtype=torch.long).to(device)
    
    prot_x = torch.rand(2, 30).to(device)
    prot_edge = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).to(device)
    prot_pos = torch.rand(2, 3).to(device)
    prot_batch = torch.tensor([0, 0], dtype=torch.long).to(device)
    
    l_batch = Batch(x=lig_x, edge_index=lig_edge, pos=lig_pos, batch=lig_batch)
    p_batch = Batch(x=prot_x, edge_index=prot_edge, pos=prot_pos, batch=prot_batch)
    l_batch.num_graphs = 1
    p_batch.num_graphs = 1
    
    with torch.no_grad():
        out = model(l_batch, p_batch)
    assert out.shape == (1,)
    assert out.device.type == "cuda"
