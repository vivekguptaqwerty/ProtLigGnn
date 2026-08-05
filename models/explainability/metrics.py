import numpy as np
import scipy.stats as stats
from typing import List, Dict, Any

def compute_rank_correlation(ranks_a: List[int], ranks_b: List[int]) -> Dict[str, float]:
    """
    Computes rank-based agreement metrics between two explanation lists of rankings:
    - Spearman Rho
    - Kendall Tau
    """
    # Align lengths
    n = min(len(ranks_a), len(ranks_b))
    if n <= 1:
        return {"spearman": 1.0, "kendall": 1.0}
        
    a = np.array(ranks_a[:n])
    b = np.array(ranks_b[:n])
    
    spearman, _ = stats.spearmanr(a, b)
    kendall, _ = stats.kendalltau(a, b)
    
    # Handle NaNs
    if np.isnan(spearman):
        spearman = 0.0
    if np.isnan(kendall):
        kendall = 0.0
        
    return {
        "spearman": float(spearman),
        "kendall": float(kendall)
    }

def compute_jaccard_similarity(rankings_a: List[int], rankings_b: List[int], k: int = 5) -> float:
    """
    Computes Jaccard Similarity between top-k elements of two rankings:
    Jaccard = Intersection(top_k_a, top_k_b) / Union(top_k_a, top_k_b)
    """
    set_a = set(rankings_a[:k])
    set_b = set(rankings_b[:k])
    
    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)
    
    if len(union) == 0:
        return 1.0
        
    return len(intersection) / len(union)

def compute_top_k_overlap(rankings_a: List[int], rankings_b: List[int], k: int = 5) -> float:
    """
    Computes direct overlap fraction between top-k elements.
    """
    set_a = set(rankings_a[:k])
    set_b = set(rankings_b[:k])
    
    intersection = set_a.intersection(set_b)
    return len(intersection) / k
