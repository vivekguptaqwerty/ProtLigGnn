import torch
from typing import Dict, Any

def compute_interaction_map_metrics(heatmap: torch.Tensor, k: int = 5) -> Dict[str, float]:
    """
    Computes diagnostic metrics for interaction heatmaps:
    - Sparsity (percentage of values below 0.1)
    - Entropy (Shannon entropy of normalized probabilities)
    - Concentration (sum of top k interaction values)
    - Top-k interaction coverage (percentage of total attention covered by top k elements)
    """
    # Normalize to probability distribution
    total_sum = heatmap.sum().item()
    if total_sum == 0:
        return {
            "sparsity": 1.0,
            "entropy": 0.0,
            "concentration": 0.0,
            "top_k_coverage": 0.0
        }
        
    p = heatmap / total_sum
    
    # 1. Sparsity
    sparsity = (heatmap < 0.1).float().mean().item()
    
    # 2. Entropy
    eps = 1e-12
    entropy = -torch.sum(p * torch.log(p + eps)).item()
    
    # 3. Concentration & Coverage
    flat_heatmap = heatmap.flatten()
    sorted_values = torch.sort(flat_heatmap, descending=True).values
    top_k_sum = sorted_values[:k].sum().item()
    top_k_coverage = top_k_sum / (total_sum + eps)
    
    return {
        "sparsity": sparsity,
        "entropy": entropy,
        "concentration": top_k_sum,
        "top_k_coverage": top_k_coverage
    }
