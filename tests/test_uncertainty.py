import pytest
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Batch, Data

# Import uncertainty framework modules
from models.uncertainty.config import (
    UncertaintyConfig, MCDropoutConfig, EnsembleConfig, EvidentialConfig, CalibrationConfig
)
from models.uncertainty.interface import PredictionWithUncertainty
from models.uncertainty.calibration import (
    calibrate_variance, compute_confidence, compute_prediction_interval
)
from models.uncertainty.calibration_metrics import (
    compute_regression_ece, compute_adaptive_ece, compute_ence,
    compute_reliability_metrics, evaluate_selective_prediction
)
from models.uncertainty.model import UncertaintyMultimodalGNN
from models.uncertainty.mc_dropout import MCDropoutEstimator
from models.uncertainty.ensemble import DeepEnsembleEstimator
from models.uncertainty.evidential import EvidentialRegressionEstimator, evidential_regression_loss

class MockGNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ligand_encoder = lambda x, edge_index: torch.zeros(x.size(0), 8)
        self.protein_encoder = lambda x, edge_index: torch.zeros(x.size(0), 8)
        self.cross_graph_interaction = lambda l_x, p_x, ligand_pos, protein_pos: (
            torch.zeros(l_x.size(0), 8), torch.zeros(p_x.size(0), 8)
        )
        self.dropout = nn.Dropout(0.1)
        self.pooling = "mean"
        self.regressor = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4, 1)
        )

class MockBaseModel(nn.Module):
    def __init__(self, output_val=5.0):
        super().__init__()
        self.base_model = MockGNNModel()
        self.regressor = self.base_model.regressor
        self.strategy = "concatenation"
        self.fusion_p = lambda p_x, l_x: torch.zeros(p_x.size(0), 8)
        self.fusion_l = lambda l_x, p_x: torch.zeros(l_x.size(0), 8)
        self.output_val = output_val

    def forward(self, ligand_batch, protein_batch):
        # Return constant or simple function of batch sizes
        batch_size = ligand_batch.num_graphs if hasattr(ligand_batch, 'num_graphs') else 1
        return torch.full((batch_size,), self.output_val, dtype=torch.float32)

def create_mock_batches(batch_size=2):
    # Setup simple mock PyG Batch objects
    ligand_data_list = []
    protein_data_list = []
    
    for _ in range(batch_size):
        l_data = Data(
            x=torch.zeros((4, 16), dtype=torch.float32),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
            pos=torch.zeros((4, 3), dtype=torch.float32)
        )
        p_data = Data(
            x=torch.zeros((5, 16), dtype=torch.float32),
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
            pos=torch.zeros((5, 3), dtype=torch.float32)
        )
        ligand_data_list.append(l_data)
        protein_data_list.append(p_data)
        
    ligand_batch = Batch.from_data_list(ligand_data_list)
    protein_batch = Batch.from_data_list(protein_data_list)
    
    return ligand_batch, protein_batch

def test_mc_dropout_inference():
    base_model = MockBaseModel()
    mc_config = MCDropoutConfig(mc_samples=15)
    calib_config = CalibrationConfig(calibration_method="none")
    
    estimator = MCDropoutEstimator(base_model, mc_config, calib_config)
    l_batch, p_batch = create_mock_batches(batch_size=3)
    
    pred_obj = estimator.predict_with_uncertainty(l_batch, p_batch)
    
    assert isinstance(pred_obj, PredictionWithUncertainty)
    assert pred_obj.method == "mc_dropout"
    assert pred_obj.affinity.shape == (3,)
    assert torch.all(pred_obj.total_uncertainty >= 0.0)
    assert torch.all(pred_obj.confidence >= 0.0) and torch.all(pred_obj.confidence <= 1.0)
    
    lower, upper = pred_obj.prediction_interval
    assert torch.all(lower <= upper)

def test_deep_ensemble_inference():
    members = [MockBaseModel(output_val=4.5), MockBaseModel(output_val=5.5)]
    ens_config = EnsembleConfig(ensemble_size=2)
    calib_config = CalibrationConfig(calibration_method="none")
    
    estimator = DeepEnsembleEstimator(members, ens_config, calib_config)
    l_batch, p_batch = create_mock_batches(batch_size=2)
    
    pred_obj = estimator.predict_with_uncertainty(l_batch, p_batch)
    
    assert isinstance(pred_obj, PredictionWithUncertainty)
    assert pred_obj.method == "ensemble"
    assert torch.allclose(pred_obj.affinity, torch.tensor([5.0, 5.0]))
    # Expected variance: var([4.5, 5.5]) = 0.5
    assert torch.allclose(pred_obj.total_uncertainty, torch.tensor([0.5, 0.5]))
    
    # Check ensemble diversity tracking
    assert estimator.last_pairwise_disagreement > 0
    assert estimator.last_avg_cosine_similarity > 0

def test_ensemble_variance_zero():
    # If all members output identical values, variance should be zero
    members = [MockBaseModel(output_val=5.0), MockBaseModel(output_val=5.0)]
    ens_config = EnsembleConfig(ensemble_size=2)
    calib_config = CalibrationConfig(calibration_method="none")
    
    estimator = DeepEnsembleEstimator(members, ens_config, calib_config)
    l_batch, p_batch = create_mock_batches(batch_size=2)
    pred_obj = estimator.predict_with_uncertainty(l_batch, p_batch)
    
    assert torch.allclose(pred_obj.total_uncertainty, torch.zeros(2))

def test_evidential_regression_inference():
    base_model = MockBaseModel()
    ev_config = EvidentialConfig()
    calib_config = CalibrationConfig(calibration_method="none")
    
    estimator = EvidentialRegressionEstimator(base_model, ev_config, calib_config)
    l_batch, p_batch = create_mock_batches(batch_size=3)
    
    pred_obj = estimator.predict_with_uncertainty(l_batch, p_batch)
    
    assert isinstance(pred_obj, PredictionWithUncertainty)
    assert pred_obj.method == "evidential"
    assert pred_obj.affinity.shape == (3,)
    assert torch.all(pred_obj.aleatoric_uncertainty > 0.0)
    assert torch.all(pred_obj.epistemic_uncertainty > 0.0)
    assert torch.all(pred_obj.total_uncertainty > 0.0)
    
    lower, upper = pred_obj.prediction_interval
    assert torch.all(lower <= upper)

def test_evidential_loss_stability():
    # Verify evidential regression loss computes correctly and gradients flow
    gamma = torch.tensor([5.0, 6.0], requires_grad=True)
    v = torch.tensor([2.0, 3.0], requires_grad=True)
    alpha = torch.tensor([2.5, 3.5], requires_grad=True)
    beta = torch.tensor([1.2, 1.8], requires_grad=True)
    target = torch.tensor([5.2, 5.8])
    
    loss = evidential_regression_loss(gamma, v, alpha, beta, target, lambda_val=0.1)
    
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() > 0.0
    
    loss.backward()
    assert gamma.grad is not None
    assert v.grad is not None
    assert alpha.grad is not None
    assert beta.grad is not None

def test_calibration_metrics():
    # Create simple predictions/variances/targets to verify metrics functions
    predictions = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
    variances = np.array([0.1, 0.2, 0.15, 0.3, 0.25])
    targets = np.array([5.1, 5.8, 7.2, 8.5, 8.8])
    
    # 1. ECE/MCE
    ece, mce, nominals, empiricals = compute_regression_ece(predictions, variances, targets, n_bins=5)
    assert 0.0 <= ece <= 1.0
    assert 0.0 <= mce <= 1.0
    
    # 2. Adaptive ECE
    aece = compute_adaptive_ece(predictions, variances, targets, n_bins=2)
    assert aece >= 0.0
    
    # 3. ENCE
    ence = compute_ence(predictions, variances, targets, n_bins=2)
    assert ence >= 0.0
    
    # 4. Reliability dict
    rel_dict = compute_reliability_metrics(predictions, variances, targets)
    assert "nll" in rel_dict
    assert "picp" in rel_dict
    assert "mpiw" in rel_dict
    assert "sharpness" in rel_dict
    
    # 5. Selective prediction
    sel_results = evaluate_selective_prediction(predictions, variances, targets, rejection_fractions=[0.2])
    assert len(sel_results) == 1
    assert sel_results[0]["coverage"] == 0.8
    assert "rmse" in sel_results[0]

def test_uncertainty_model_wrapper():
    base_model = MockBaseModel()
    config = UncertaintyConfig(method="mc_dropout")
    wrapper = UncertaintyMultimodalGNN(config, base_model)
    
    l_batch, p_batch = create_mock_batches(batch_size=2)
    
    # Standard forward call (backward compatible)
    out_fwd = wrapper(l_batch, p_batch)
    assert out_fwd.shape == (2,)
    
    # Uncertainty prediction call
    out_unc = wrapper.predict_with_uncertainty(l_batch, p_batch)
    assert isinstance(out_unc, PredictionWithUncertainty)
    assert out_unc.method == "mc_dropout"
