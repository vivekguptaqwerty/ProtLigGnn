import torch
import pytest
from torch_geometric.data import Data, Batch
from models.geometry_interaction.config import InteractionConfig
from models.geometry_interaction.geometry_interaction_model import ProtLigGNNGeometryInteraction

def make_mock_complex_batch(num_lig_nodes: int, num_prot_nodes: int, batch_size: int = 1):
    """Utility to create a PyG Batch of complexes."""
    ligand_list = []
    protein_list = []
    
    for _ in range(batch_size):
        # Ligand: 78 features, 3D positions
        lx = torch.zeros(num_lig_nodes, 78)
        lx[:, 1] = 1.0  # Set Carbon element
        lpos = torch.randn(num_lig_nodes, 3)
        le = torch.randint(0, num_lig_nodes, (2, num_lig_nodes * 2)) if num_lig_nodes > 1 else torch.empty((2, 0), dtype=torch.long)
        ligand_list.append(Data(x=lx, edge_index=le, pos=lpos))
        
        # Protein: 30 features, 3D positions
        px = torch.zeros(num_prot_nodes, 30)
        ppos = torch.randn(num_prot_nodes, 3)
        pe = torch.randint(0, num_prot_nodes, (2, num_prot_nodes * 2)) if num_prot_nodes > 1 else torch.empty((2, 0), dtype=torch.long)
        protein_list.append(Data(x=px, edge_index=pe, pos=ppos))
        
    return Batch.from_data_list(ligand_list), Batch.from_data_list(protein_list)


def test_geometry_interaction_model_flow():
    """Verify that the model wrapper executes correctly and outputs expected shape."""
    config = InteractionConfig(interaction_dim=16, latent_dim=16)
    model = ProtLigGNNGeometryInteraction(ligand_dim=78, protein_dim=30, hidden_dim=64, config=config)
    model.eval()
    
    l_batch, p_batch = make_mock_complex_batch(10, 15, batch_size=2)
    with torch.no_grad():
        out = model(l_batch, p_batch)
        
    assert out.shape == (2,)
    assert out.dtype == torch.float32


def test_gradient_flow():
    """Verify that gradients propagate to GNN encoders and all branches of the IRM."""
    config = InteractionConfig(interaction_dim=16, latent_dim=16)
    model = ProtLigGNNGeometryInteraction(ligand_dim=78, protein_dim=30, hidden_dim=64, config=config)
    model.train()
    
    l_batch, p_batch = make_mock_complex_batch(8, 12, batch_size=1)
    
    out = model(l_batch, p_batch)
    loss = out.sum()
    loss.backward()
    
    # 1. Check GNN encoders gradients
    for name, param in model.ligand_encoder.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        
    # 2. Check Geometry, Chemistry, and Representation branches gradients
    attn_bias = model.cross_attention.attention_bias
    for name, param in attn_bias.geometry_encoder.named_parameters():
        assert param.grad is not None, f"No gradient for geometry encoder {name}"
    for name, param in attn_bias.chemistry_encoder.named_parameters():
        assert param.grad is not None, f"No gradient for chemistry encoder {name}"
    for name, param in attn_bias.representation_encoder.named_parameters():
        assert param.grad is not None, f"No gradient for representation encoder {name}"
    for name, param in attn_bias.fusion.named_parameters():
        assert param.grad is not None, f"No gradient for fusion {name}"


def test_determinism():
    """Verify that model execution is fully deterministic."""
    torch.manual_seed(42)
    config = InteractionConfig()
    model = ProtLigGNNGeometryInteraction(ligand_dim=78, protein_dim=30, hidden_dim=64, config=config)
    model.eval()
    
    l_batch, p_batch = make_mock_complex_batch(10, 10, batch_size=1)
    with torch.no_grad():
        out1 = model(l_batch, p_batch)
        out2 = model(l_batch, p_batch)
        
    assert torch.allclose(out1, out2)


def test_gpu_compatibility():
    """Verify that model executes correctly on CUDA if available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available.")
        
    device = torch.device("cuda")
    config = InteractionConfig()
    model = ProtLigGNNGeometryInteraction(ligand_dim=78, protein_dim=30, hidden_dim=64, config=config).to(device)
    model.eval()
    
    l_batch, p_batch = make_mock_complex_batch(10, 10, batch_size=1)
    l_batch = l_batch.to(device)
    p_batch = p_batch.to(device)
    
    with torch.no_grad():
        out = model(l_batch, p_batch)
        
    assert out.device.type == "cuda"


def test_extreme_edge_cases():
    """Verify that the model handles disconnected, empty edge lists, and tiny graphs gracefully."""
    config = InteractionConfig()
    model = ProtLigGNNGeometryInteraction(ligand_dim=78, protein_dim=30, hidden_dim=64, config=config)
    model.eval()
    
    # 1. 1-node, 0-edge empty graphs (lowest possible graph sizes)
    l_batch, p_batch = make_mock_complex_batch(1, 1, batch_size=1)
    with torch.no_grad():
        out = model(l_batch, p_batch)
    assert out.shape == (1,)

    # 2. Fully disconnected graphs (nodes have 0 edges)
    l_batch.edge_index = torch.empty((2, 0), dtype=torch.long)
    p_batch.edge_index = torch.empty((2, 0), dtype=torch.long)
    with torch.no_grad():
        out_disconnected = model(l_batch, p_batch)
    assert out_disconnected.shape == (1,)


def test_permutation_consistency():
    """Verify that permuting ligand/protein node order yields identical predictions."""
    config = InteractionConfig()
    model = ProtLigGNNGeometryInteraction(ligand_dim=78, protein_dim=30, hidden_dim=64, config=config)
    model.eval()
    
    # 1. Base batch
    l_batch, p_batch = make_mock_complex_batch(6, 8, batch_size=1)
    
    # Get original prediction
    with torch.no_grad():
        pred_orig = model(l_batch, p_batch)
        
    # 2. Permute ligand nodes
    perm_l = torch.randperm(6)
    l_batch_perm = l_batch.clone()
    l_batch_perm.x = l_batch.x[perm_l]
    l_batch_perm.pos = l_batch.pos[perm_l]
    
    # Adjust edge index to match permuted node indices
    inv_perm_l = torch.zeros(6, dtype=torch.long)
    inv_perm_l[perm_l] = torch.arange(6)
    l_batch_perm.edge_index = inv_perm_l[l_batch.edge_index]
    
    with torch.no_grad():
        pred_perm = model(l_batch_perm, p_batch)
        
    assert torch.allclose(pred_orig, pred_perm, atol=1e-5)


def test_batched_inference_consistency():
    """Verify that predictions in a batch match individual predictions."""
    config = InteractionConfig()
    model = ProtLigGNNGeometryInteraction(ligand_dim=78, protein_dim=30, hidden_dim=64, config=config)
    model.eval()
    
    l_b1, p_b1 = make_mock_complex_batch(5, 7, batch_size=1)
    l_b2, p_b2 = make_mock_complex_batch(6, 8, batch_size=1)
    
    # Individual runs
    with torch.no_grad():
        out1 = model(l_b1, p_b1)
        out2 = model(l_b2, p_b2)
        
    # Batched run
    l_batch = Batch.from_data_list([l_b1.to_data_list()[0], l_b2.to_data_list()[0]])
    p_batch = Batch.from_data_list([p_b1.to_data_list()[0], p_b2.to_data_list()[0]])
    
    with torch.no_grad():
        out_batch = model(l_batch, p_batch)
        
    assert torch.allclose(out_batch[0], out1[0], atol=1e-5)
    assert torch.allclose(out_batch[1], out2[0], atol=1e-5)


def test_cross_device_consistency():
    """Verify that CPU and GPU outputs are numerically consistent."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available.")
        
    config = InteractionConfig()
    model_cpu = ProtLigGNNGeometryInteraction(ligand_dim=78, protein_dim=30, hidden_dim=64, config=config)
    model_cpu.eval()
    
    device = torch.device("cuda")
    model_gpu = ProtLigGNNGeometryInteraction(ligand_dim=78, protein_dim=30, hidden_dim=64, config=config).to(device)
    model_gpu.load_state_dict(model_cpu.state_dict())
    model_gpu.eval()
    
    import copy
    l_batch_cpu, p_batch_cpu = make_mock_complex_batch(6, 8, batch_size=1)
    l_batch_gpu = copy.deepcopy(l_batch_cpu).to(device)
    p_batch_gpu = copy.deepcopy(p_batch_cpu).to(device)
    
    with torch.no_grad():
        out_cpu = model_cpu(l_batch_cpu, p_batch_cpu)
        out_gpu = model_gpu(l_batch_gpu, p_batch_gpu)
        
    assert torch.allclose(out_cpu, out_gpu.cpu(), atol=1e-5)


def test_seed_reproducibility():
    """Verify that setting the random seed ensures deterministic initialization and outputs."""
    torch.manual_seed(42)
    model1 = ProtLigGNNGeometryInteraction(ligand_dim=78, protein_dim=30, hidden_dim=64)
    
    torch.manual_seed(42)
    model2 = ProtLigGNNGeometryInteraction(ligand_dim=78, protein_dim=30, hidden_dim=64)
    
    for w1, w2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(w1, w2)

