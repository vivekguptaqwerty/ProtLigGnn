import pytest
import torch
import numpy as np
import scipy.stats as stats

def compute_entropy(attn: np.ndarray, eps: float = 1e-12) -> float:
    """Computes normalized attention entropy."""
    L, P = attn.shape
    return float(-np.sum(attn * np.log(attn + eps)) / (L * P))

def test_metrics_calculation() -> None:
    """Tests the mathematical calculation of entropy and distance correlation."""
    # 2 query nodes, 2 key nodes
    # Mock attention weights (must sum to 1 per row for a valid attention map)
    # L=2, P=2
    attn = np.array([[0.8, 0.2], [0.1, 0.9]])
    
    # Pairwise distances: atom 0 is close to residue 0 (dist=1.0) and far from residue 1 (dist=5.0)
    # atom 1 is far from residue 0 (dist=5.0) and close to residue 1 (dist=1.0)
    dists = np.array([[1.0, 5.0], [5.0, 1.0]])
    inv_dists = 1.0 / dists
    
    # Pearson correlation
    pearson_r, _ = stats.pearsonr(attn.flatten(), inv_dists.flatten())
    # Should be positive because high attention corresponds to small distance (high inv_dist)
    assert pearson_r > 0.5
    
    # Entropy calculation
    entropy = compute_entropy(attn)
    # Entropy should be bounded
    assert entropy > 0.0
    assert entropy < 1.0
    
    # Test flat or identical maps (entropy should be maximized)
    flat_attn = np.array([[0.5, 0.5], [0.5, 0.5]])
    flat_entropy = compute_entropy(flat_attn)
    assert flat_entropy > entropy  # Flat attention is more diffuse, higher entropy
