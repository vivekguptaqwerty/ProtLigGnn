import pytest
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Batch, Data

# Import explainability framework modules
from models.explainability.config import (
    ExplainabilityConfig, IntegratedGradientsConfig, AttentionRolloutConfig, OcclusionConfig, GraphAttributionConfig
)
from models.explainability.interface import ExplanationResult, ExplanationEngine
from models.explainability.integrated_gradients import IntegratedGradientsEngine
from models.explainability.attention_rollout import AttentionRolloutEngine
from models.explainability.occlusion import OcclusionEngine
from models.explainability.graph_explainer import GraphExplainerEngine
from models.explainability.faithfulness import compute_faithfulness_curves, compute_infidelity_and_sensitivity
from models.explainability.metrics import compute_rank_correlation, compute_jaccard_similarity, compute_top_k_overlap
from models.explainability.interaction_maps import compute_interaction_map_metrics

# Re-use MockGNNModel and MockBaseModel from test_uncertainty
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
        
        # Add mock cross_attention object with MultiheadAttention sub-modules for hook tests
        class MockCrossAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.ligand_to_protein_attn = nn.MultiheadAttention(8, 2, batch_first=True)
                self.protein_to_ligand_attn = nn.MultiheadAttention(8, 2, batch_first=True)
        self.cross_attention = MockCrossAttention()

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
        batch_size = ligand_batch.num_graphs if hasattr(ligand_batch, 'num_graphs') else 1
        # Sum coordinate tensor to make gradients mathematically non-zero for tests
        pos_sum = ligand_batch.pos.sum() + protein_batch.pos.sum()
        # Return scalar with position contribution
        return torch.full((batch_size,), self.output_val, dtype=torch.float32) + 0.01 * pos_sum

def create_mock_batches(batch_size=2):
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

def test_integrated_gradients_correctness():
    base_model = MockBaseModel()
    ig_config = IntegratedGradientsConfig(steps=10)
    engine = IntegratedGradientsEngine(base_model, ig_config)
    
    l_batch, p_batch = create_mock_batches(batch_size=2)
    res = engine.explain(l_batch, p_batch)
    
    assert isinstance(res, ExplanationResult)
    assert res.explanation_method == "integrated_gradients"
    assert res.atom_scores.shape == (8,)  # 2 graphs * 4 nodes
    assert res.residue_scores.shape == (10,)  # 2 graphs * 5 nodes
    
    # Check completeness axiom error is recorded
    assert res.explanation_confidence.item() >= 0.0

def test_attention_rollout_hook():
    base_model = MockBaseModel()
    attn_config = AttentionRolloutConfig()
    engine = AttentionRolloutEngine(base_model, attn_config)
    
    l_batch, p_batch = create_mock_batches(batch_size=2)
    
    # Run explain (triggers forward pass and hooks)
    res = engine.explain(l_batch, p_batch)
    
    assert isinstance(res, ExplanationResult)
    assert res.explanation_method == "attention_rollout"
    assert res.atom_scores.shape == (8,)
    assert res.residue_scores.shape == (10,)

def test_occlusion_nan_and_bounds():
    base_model = MockBaseModel()
    occ_config = OcclusionConfig()
    engine = OcclusionEngine(base_model, occ_config)
    
    l_batch, p_batch = create_mock_batches(batch_size=1)
    res = engine.explain(l_batch, p_batch)
    
    assert isinstance(res, ExplanationResult)
    assert res.explanation_method == "occlusion"
    assert not torch.isnan(res.atom_scores).any()
    assert not torch.isnan(res.residue_scores).any()
    assert torch.all(res.atom_scores >= 0.0)
    assert torch.all(res.residue_scores >= 0.0)

def test_graph_attribution():
    base_model = MockBaseModel()
    graph_config = GraphAttributionConfig()
    engine = GraphExplainerEngine(base_model, graph_config)
    
    l_batch, p_batch = create_mock_batches(batch_size=1)
    res = engine.explain(l_batch, p_batch)
    
    assert isinstance(res, ExplanationResult)
    assert res.explanation_method == "graph_attribution"
    assert torch.all(res.atom_scores <= 1.0)
    assert torch.all(res.residue_scores <= 1.0)

def test_faithfulness_metrics_auc():
    base_model = MockBaseModel()
    l_batch, p_batch = create_mock_batches(batch_size=1)
    
    atom_scores = torch.rand(4)
    residue_scores = torch.rand(5)
    
    curves = compute_faithfulness_curves(base_model, l_batch, p_batch, atom_scores, residue_scores, steps=5)
    
    assert "deletion_auc" in curves
    assert "insertion_auc" in curves
    assert len(curves["deletion_curve"]) == 6
    assert len(curves["insertion_curve"]) == 6
    assert 0.0 <= curves["deletion_auc"] <= 10.0
    assert 0.0 <= curves["insertion_auc"] <= 10.0

def test_agreement_metrics():
    rankings_a = [0, 1, 2, 3, 4]
    rankings_b = [1, 0, 2, 4, 3]
    
    rank_corr = compute_rank_correlation(rankings_a, rankings_b)
    jaccard = compute_jaccard_similarity(rankings_a, rankings_b, k=3)
    top_k = compute_top_k_overlap(rankings_a, rankings_b, k=3)
    
    assert "spearman" in rank_corr
    assert "kendall" in rank_corr
    assert 0.0 <= jaccard <= 1.0
    assert 0.0 <= top_k <= 1.0

def test_heatmap_diagnostics():
    heatmap = torch.rand((5, 4))
    metrics = compute_interaction_map_metrics(heatmap, k=3)
    
    assert "sparsity" in metrics
    assert "entropy" in metrics
    assert "concentration" in metrics
    assert "top_k_coverage" in metrics
    assert metrics["sparsity"] >= 0.0
    assert metrics["top_k_coverage"] >= 0.0
