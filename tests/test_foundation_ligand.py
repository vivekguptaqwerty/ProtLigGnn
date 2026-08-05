import tempfile
import torch
import pytest
from pathlib import Path
from torch_geometric.data import Data, Batch

from models.foundation.ligand.config import LigandFoundationConfig
from models.foundation.ligand.encoders import ChemBERTaEncoder, MolFormerEncoder
from models.foundation.ligand.alignment import align_ligand_tokens, canonicalize_smiles_safe
from models.foundation.ligand.cache import load_ligand_with_cache, get_tensor_checksum
from models.foundation.ligand.model import HybridLigandGNN, LigandFoundationOnly

def test_config_validation():
    cfg = LigandFoundationConfig(ligand_model="chemberta")
    assert cfg.projection_dimension == 256
    
    with pytest.raises(ValueError):
        LigandFoundationConfig(ligand_model="invalid_model")

def test_canonicalize_smiles():
    raw = "C1=CC=CC=C1" # benzene
    canonical, diag = canonicalize_smiles_safe(raw)
    assert canonical == "c1ccccc1"
    assert not diag["invalid_molecule"]
    
    invalid, diag_inv = canonicalize_smiles_safe("invalid_smiles_string")
    assert invalid == ""
    assert diag_inv["invalid_molecule"]

def test_alignment_mapping():
    smiles = "CCO" # Ethanol: C (0), C (1), O (2)
    tokens = ["[CLS]", "c", "c", "o", "[SEP]"]
    mapping, stats = align_ligand_tokens(smiles, tokens)
    
    assert len(mapping) == 3
    assert mapping[0] == 1 # First Carbon matches token index 1
    assert mapping[1] == 2 # Second Carbon matches token index 2
    assert mapping[2] == 3 # Oxygen matches token index 3
    assert stats["matched_atoms"] == 3
    assert stats["unmatched_atoms"] == 0

def test_encoders_and_emulator():
    cfg = LigandFoundationConfig(ligand_model="chemberta")
    encoder = ChemBERTaEncoder(cfg, device="cpu")
    
    assert encoder.emulator_active
    res = encoder.encode("CCO")
    assert "atom_embeddings" in res
    assert "molecule_embedding" in res
    assert res["atom_embeddings"].shape[1] == 384
    assert res["is_emulator"].item() is True

def test_cache_validation_and_checksum():
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg = LigandFoundationConfig(ligand_model="chemberta", cache_directory=tmp_dir)
        encoder = ChemBERTaEncoder(cfg, device="cpu")
        
        # Initial save
        pdb_id = "test_complex"
        smiles = "CCO"
        atom_embs, mol_emb, is_em = load_ligand_with_cache(pdb_id, smiles, cfg, encoder)
        
        # Verify cached files exist
        pt_path = Path(tmp_dir) / f"{pdb_id}_embedding.pt"
        json_path = Path(tmp_dir) / f"{pdb_id}_metadata.json"
        assert pt_path.exists()
        assert json_path.exists()
        
        # Reload and check validation passes
        atom_embs_r, mol_emb_r, is_em_r = load_ligand_with_cache(pdb_id, smiles, cfg, encoder)
        assert torch.equal(atom_embs, atom_embs_r)
        assert torch.equal(mol_emb, mol_emb_r)

def test_hybrid_and_ablation_model():
    cfg = LigandFoundationConfig(ligand_model="chemberta")
    
    # Mock base GNN
    class MockBaseGNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ligand_encoder = lambda x, edge_index: x
            self.protein_encoder = lambda x, edge_index: x
            self.cross_graph_interaction = lambda lx, px: (lx, px)
            self.pooling = "mean"
            self.dropout = torch.nn.Dropout(0.1)
            self.regressor = torch.nn.Linear(512, 1)
            
    base_gnn = MockBaseGNN()
    model = HybridLigandGNN(base_hybrid=base_gnn, config=cfg)
    
    # Mock data batches
    l_x = torch.randn(4, 256)
    l_pos = torch.randn(4, 3)
    l_edge = torch.tensor([[0, 1, 2], [1, 2, 3]])
    l_batch = torch.tensor([0, 0, 0, 0])
    l_found = torch.randn(4, 384)
    
    ligand_batch = Batch(x=l_x, pos=l_pos, edge_index=l_edge, batch=l_batch, num_graphs=1, foundation_emb=l_found)
    
    p_x = torch.randn(6, 256)
    p_pos = torch.randn(6, 3)
    p_edge = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    p_batch = torch.tensor([0, 0, 0, 0, 0, 0])
    
    protein_batch = Batch(x=p_x, pos=p_pos, edge_index=p_edge, batch=p_batch, num_graphs=1)
    
    # Forward pass and gradient verification
    out = model(ligand_batch, protein_batch)
    assert out.shape == (1,)
    
    loss = out.sum()
    loss.backward()
    
    # Verify gradient flow through projection
    assert model.projection.mlp[0].weight.grad is not None
