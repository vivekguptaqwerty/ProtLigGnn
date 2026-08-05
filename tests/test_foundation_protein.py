import os
import tempfile
import json
import time
from pathlib import Path
import pytest
import torch
from torch_geometric.data import Data, Batch

from models.foundation.protein.config import ProteinFoundationConfig
from models.foundation.protein.encoders import ESM2Encoder, ProtT5Encoder, load_with_cache, get_tensor_checksum
from models.foundation.alignment import get_pocket_residues, extract_and_align_protein
from models.foundation.projection import ProjectionMLP
from models.foundation.protein.model import HybridProteinGNN, ProteinFoundationOnly

def test_config_validation():
    # Test correct config post_init validations
    config = ProteinFoundationConfig(protein_model="esm2")
    assert config.protein_model == "esm2"
    
    with pytest.raises(ValueError):
        ProteinFoundationConfig(protein_model="invalid_model")
        
    with pytest.raises(ValueError):
        ProteinFoundationConfig(layer_selection="invalid_layer")
        
    with pytest.raises(ValueError):
        ProteinFoundationConfig(pooling_strategy="invalid_pooling")

def test_projection_mlp_dimensions():
    # Input 320 (ESM-2 dim), output 256
    mlp = ProjectionMLP(input_dim=320, output_dim=256)
    x = torch.randn(5, 320)
    out = mlp(x)
    assert out.shape == (5, 256)

def test_alignment_correctness():
    # Simple check on extract_and_align_protein using a dummy pdb logic
    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = Path(tmpdir) / "test.pdb"
        # Write a mock PDB format file
        with open(pdb_path, "w") as f:
            f.write("ATOM      1  CA  ALA A   1      11.182  15.111  22.222  1.00 20.00           C\n")
            f.write("ATOM      2  CA  ARG A   2      12.182  16.111  23.222  1.00 20.00           C\n")
            f.write("ATOM      3  CA  TRP A   3      13.182  17.111  24.222  1.00 20.00           C\n")
            
        pocket_res = [("A", 2, "ARG")]
        seq, mapping, stats = extract_and_align_protein(pdb_path, pocket_res)
        
        assert seq == "ARW"
        assert mapping == [1]
        assert stats["total_protein_residues"] == 3
        assert stats["matched_residues"] == 1
        assert stats["unmatched_residues"] == 0

def test_cache_validation_and_emulator():
    # Setup temporary directory for cache testing
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ProteinFoundationConfig(
            protein_model="esm2",
            cache_directory=tmpdir,
            projection_dimension=256
        )
        # Set emulator mode
        os.environ["LIGPROTGNN_PRODUCTION"] = "0"
        encoder = ESM2Encoder(config=config, device="cpu")
        
        sequence = "MGAARGTRW"
        pdb_id = "test_pdb"
        
        # First extraction (cache miss)
        res_embs, seq_emb, is_emulator = load_with_cache(pdb_id, sequence, config, encoder)
        assert is_emulator is True
        assert res_embs.shape == (len(sequence), 320)
        
        # Validate that cache files were created
        pt_file = Path(tmpdir) / f"{pdb_id}_embedding.pt"
        json_file = Path(tmpdir) / f"{pdb_id}_metadata.json"
        assert pt_file.exists()
        assert json_file.exists()
        
        # Second extraction (cache hit)
        res_embs_2, seq_emb_2, is_emulator_2 = load_with_cache(pdb_id, sequence, config, encoder)
        assert torch.allclose(res_embs, res_embs_2)
        assert is_emulator_2 is True

def test_hybrid_model_forward_and_gradients():
    # Test the forward execution and gradient propagation
    config = ProteinFoundationConfig(
        protein_model="esm2",
        projection_dimension=128
    )
    
    # Mock base GNN
    class MockEncoder(torch.nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.linear = torch.nn.Linear(in_dim, out_dim)
        def forward(self, x, edge_index):
            return self.linear(x)
            
    class MockGNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ligand_encoder = MockEncoder(9, 128)
            self.protein_encoder = MockEncoder(30, 128)
            self.pooling = "mean"
            self.dropout = torch.nn.Dropout(0.1)
            self.regressor = torch.nn.Linear(256, 1)
            
        def cross_graph_interaction(self, l_chunk, p_chunk):
            return l_chunk, p_chunk
            
    base_gnn = MockGNN()
    hybrid_model = HybridProteinGNN(base_gnn=base_gnn, config=config)
    
    # Construct mock PyG batches
    ligand_data = Data(x=torch.randn(4, 9), edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long))
    ligand_batch = Batch.from_data_list([ligand_data])
    
    protein_data = Data(
        x=torch.randn(6, 30),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
        foundation_emb=torch.randn(6, 320)  # Pre-attached foundation embeddings
    )
    protein_batch = Batch.from_data_list([protein_data])
    
    # Forward step
    output = hybrid_model(ligand_batch, protein_batch)
    assert output.shape == (1,)
    
    # Backward step (Gradient flow test)
    loss = output.sum()
    loss.backward()
    
    # Check that gradients flow to projection head
    for name, param in hybrid_model.projection.named_parameters():
        assert param.grad is not None
