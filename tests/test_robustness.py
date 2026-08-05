import pytest
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Batch, Data

# Import robustness framework modules
from models.robustness.config import RobustnessConfig, PerturbationConfig
from models.robustness.interface import RobustnessResult, RobustnessEvaluator
from models.robustness.robustness_suite import UncertaintyRobustnessEvaluator
from models.robustness.coordinate_noise import inject_coordinate_noise
from models.robustness.feature_noise import inject_feature_noise
from models.robustness.masking import apply_node_masking
from models.robustness.adversarial import generate_adversarial_coordinates
from models.robustness.metrics import compute_composite_robustness_index, fit_noise_response_curve, compute_explanation_drift

# Re-use MockGNNModel and MockBaseModel from test_explainability
class MockGNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        class MockEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(16, 8)
            def forward(self, x, edge_index):
                return self.linear(x)
                
        self.ligand_encoder = MockEncoder()
        self.protein_encoder = MockEncoder()
        self.cross_graph_interaction = lambda l_x, p_x, ligand_pos, protein_pos: (
            l_x, p_x
        )
        self.dropout = nn.Dropout(0.1)
        self.pooling = "mean"
        self.regressor = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        
class MockBaseModel(nn.Module):
    def __init__(self, output_val=5.0):
        super().__init__()
        self.base_model = MockGNNModel()
        self.regressor = self.base_model.regressor
        self.output_val = output_val

    def forward(self, ligand_batch, protein_batch):
        batch_size = ligand_batch.num_graphs if hasattr(ligand_batch, 'num_graphs') else 1
        pos_sum = ligand_batch.pos.sum() + protein_batch.pos.sum()
        return torch.full((batch_size,), self.output_val, dtype=torch.float32) + 0.01 * pos_sum

def create_mock_batches(batch_size=2):
    ligand_data_list = []
    protein_data_list = []
    
    for _ in range(batch_size):
        l_data = Data(
            x=torch.zeros((4, 16), dtype=torch.float32),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
            pos=torch.ones((4, 3), dtype=torch.float32)
        )
        p_data = Data(
            x=torch.zeros((5, 16), dtype=torch.float32),
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
            pos=torch.ones((5, 3), dtype=torch.float32)
        )
        ligand_data_list.append(l_data)
        protein_data_list.append(p_data)
        
    ligand_batch = Batch.from_data_list(ligand_data_list)
    protein_batch = Batch.from_data_list(protein_data_list)
    
    return ligand_batch, protein_batch

def test_coordinate_noise_correctness():
    l_batch, _ = create_mock_batches(batch_size=1)
    
    # 1. Zero-noise should be strict identity operation
    l_perturbed_zero = inject_coordinate_noise(l_batch, sigma=0.0)
    assert torch.equal(l_batch.pos, l_perturbed_zero.pos)
    
    # 2. Non-zero noise should change coordinates
    l_perturbed_noise = inject_coordinate_noise(l_batch, sigma=0.2)
    assert not torch.equal(l_batch.pos, l_perturbed_noise.pos)
    assert l_perturbed_noise.pos.shape == l_batch.pos.shape

def test_feature_noise_and_shape():
    l_batch, _ = create_mock_batches(batch_size=1)
    # Give indices
    l_batch.x = torch.randint(0, 10, (4, 16), dtype=torch.long)
    
    l_pert = inject_feature_noise(l_batch, fraction=0.2)
    assert l_pert.x.shape == l_batch.x.shape

def test_node_masking():
    l_batch, _ = create_mock_batches(batch_size=1)
    l_masked = apply_node_masking(l_batch, fraction=0.5)
    
    assert l_masked.x.shape == l_batch.x.shape

def test_adversarial_coordinate_generation():
    base_model = MockBaseModel()
    l_batch, p_batch = create_mock_batches(batch_size=1)
    
    l_adv = generate_adversarial_coordinates(base_model, l_batch, p_batch, epsilon=0.01, steps=2)
    
    assert l_adv.pos.shape == l_batch.pos.shape
    # Check that positions are perturbed
    assert not torch.equal(l_adv.pos, l_batch.pos)

def test_robustness_metrics():
    # Test composite index
    weights = {"prediction_stability": 0.3, "calibration_stability": 0.2, "explanation_stability": 0.2, "representation_stability": 0.1, "ood_performance": 0.2}
    cri = compute_composite_robustness_index(weights, 0.9, 0.95, 0.88, 0.92, 0.94)
    assert 0.8 <= cri <= 1.0
    
    # Test curve fitting
    noise_levels = [0.0, 0.1, 0.2, 0.5]
    metric_vals = [1.0, 0.9, 0.7, 0.4]
    curve = fit_noise_response_curve(noise_levels, metric_vals)
    assert "audc" in curve
    assert "slope" in curve
    assert "half_performance_threshold" in curve
    assert curve["half_performance_threshold"] >= 0.0

def test_evaluator_nominals():
    base_model = MockBaseModel()
    config = RobustnessConfig()
    evaluator = UncertaintyRobustnessEvaluator(base_model, config)
    
    l_batch, p_batch = create_mock_batches(batch_size=1)
    res = evaluator.evaluate(l_batch, p_batch)
    
    assert isinstance(res, RobustnessResult)
    assert res.perturbation_type == "none"
    assert res.robustness_score == 0.9412
