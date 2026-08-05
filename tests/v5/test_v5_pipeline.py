import os
import pytest
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from ligprotgnnx.v5.esm2_encoder import ESM2FeatureExtractor
from ligprotgnnx.v5.physical_features import BiophysicalFeatureExtractor, BiophysicalFeatureEncoder
from ligprotgnnx.v5.tensor_egnn import TensorEGNNLayer, TensorEGNN
from ligprotgnnx.v5.tta_ensemble import TestTimeAugmentationEngine
from ligprotgnnx.v5.v5_model import LigProtGNNXv5Model

def test_esm2_feature_extractor():
    cache_dir = "scratch/test_tmp/esm2_cache"
    os.makedirs(cache_dir, exist_ok=True)
    extractor = ESM2FeatureExtractor(cache_dir=cache_dir)
    seq = "ACDEFGHIKLMNPQRSTVWY"
    embeddings = extractor.extract_sequence_embedding(seq)
    assert embeddings.shape == (20, 1280), f"Expected (20, 1280), got {embeddings.shape}"
    
    # Verify fallback forward pass
    one_hots = torch.eye(20)
    out = extractor(one_hots)
    assert out.shape == (20, 1280)

def test_biophysical_features():
    mol = Chem.MolFromSmiles("CC(=O)O")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    
    charges = BiophysicalFeatureExtractor.compute_gasteiger_charges(mol)
    assert len(charges) > 0
    
    coords = np.random.randn(5, 3).astype(np.float32)
    sasa = BiophysicalFeatureExtractor.compute_sasa_estimate(coords)
    assert len(sasa) == 5
    assert np.all(sasa >= 0.0)

def test_tensor_egnn_equivariance():
    layer = TensorEGNNLayer(node_dim=64, edge_dim=16, hidden_dim=64)
    layer.eval()
    
    N = 10
    h = torch.randn(N, 64)
    x = torch.randn(N, 3)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7],
                               [1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    edge_attr = torch.randn(8, 16)
    
    h1, x1 = layer(h, x, edge_index, edge_attr)
    
    # Apply random 3D rotation matrix R in SO(3)
    theta = np.pi / 4
    R = torch.tensor([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0]
    ], dtype=torch.float32)
    
    x_rotated = torch.matmul(x, R.T)
    h2, x2 = layer(h, x_rotated, edge_index, edge_attr)
    
    # Verify node scalar features are invariant
    assert torch.allclose(h1, h2, atol=1e-4), "Node scalar features must be rotationally invariant!"
    
    # Verify node spatial coordinates are equivariant: R(x1) == x2
    x1_rotated = torch.matmul(x1, R.T)
    assert torch.allclose(x1_rotated, x2, atol=1e-4), "Coordinates must update equivariantly under 3D rotation!"

def test_tta_ensemble():
    engine = TestTimeAugmentationEngine(num_augmentations=3, jitter_std=0.05)
    coords = torch.zeros(10, 3)
    coords_jittered = engine.apply_spatial_jitter(coords)
    assert coords_jittered.shape == coords.shape
    assert not torch.allclose(coords, coords_jittered)

def test_v5_model_forward():
    model = LigProtGNNXv5Model(hidden_dim=64, rbf_dim=16, num_egnn_layers=2)
    model.eval()
    
    class DummyData:
        pass
        
    data = DummyData()
    data.x_protein = torch.eye(20)
    data.pos_protein = torch.randn(20, 3)
    data.edge_index_protein = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    
    data.x_ligand = torch.eye(35)
    data.pos_ligand = torch.randn(35, 3)
    data.edge_index_ligand = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    
    pred_affinity, epistemic_var = model(data)
    assert pred_affinity.shape[0] == 1
    assert epistemic_var.shape[0] == 1
    assert epistemic_var.item() > 0.0
