import numpy as np
import pytest
from models.physics_guided.metrics import compute_contact_metrics, compute_localization_metrics

def test_compute_contact_metrics():
    # Simple binary outcomes
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred_prob = np.array([0.9, 0.1, 0.8, 0.4, 0.2])  # threshold >= 0.5 gives [1, 0, 1, 0, 0]
    
    metrics = compute_contact_metrics(y_true, y_pred_prob)
    
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "balanced_accuracy" in metrics
    assert "mcc" in metrics
    assert "brier_score" in metrics
    
    # Check simple values
    # True positives: 2 (indices 0, 2), False positives: 0, False negatives: 1 (index 3), True negatives: 2 (indices 1, 4)
    # Precision = 2 / 2 = 1.0
    # Recall = 2 / 3 = 0.666...
    assert abs(metrics["precision"] - 1.0) < 1e-5
    assert abs(metrics["recall"] - 2.0/3.0) < 1e-5

def test_compute_localization_metrics():
    attn = np.array([0.5, 0.3, 0.2])
    dists = np.array([2.0, 5.0, 8.0])  # threshold < 4.0 leaves only index 0 (1 contact)
    
    loc = compute_localization_metrics(attn, dists, threshold=4.0)
    
    assert "attention_entropy" in loc
    assert "distance_correlation" in loc
    assert "contact_recall" in loc
    assert "topk_localization" in loc
    
    # 1 contact, so top-1 attn is index 0, which is indeed contact -> 1.0
    assert abs(loc["contact_recall"] - 1.0) < 1e-5
