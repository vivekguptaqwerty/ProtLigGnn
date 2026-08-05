import torch
import pytest
from models.physics_guided.contact_head import ContactPredictionHead

def test_contact_head_shapes():
    L, P, D = 10, 15, 64
    ligand_nodes = torch.randn(L, D)
    protein_nodes = torch.randn(P, D)
    
    head = ContactPredictionHead(node_dim=D, hidden_dim=128, activation="relu", normalization="layer")
    logits = head(ligand_nodes, protein_nodes)
    
    assert logits.shape == (L, P)

def test_contact_head_no_normalization():
    L, P, D = 5, 8, 32
    ligand_nodes = torch.randn(L, D)
    protein_nodes = torch.randn(P, D)
    
    head = ContactPredictionHead(node_dim=D, hidden_dim=64, activation="gelu", normalization="none")
    logits = head(ligand_nodes, protein_nodes)
    
    assert logits.shape == (L, P)
