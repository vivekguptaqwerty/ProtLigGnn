import numpy as np
import scipy.stats as stats
from typing import Dict, List, Any

def compute_composite_robustness_index(
    weights: Dict[str, float],
    prediction_stability: float,
    calibration_stability: float,
    explanation_stability: float,
    representation_stability: float,
    ood_performance: float
) -> float:
    """
    Computes the Composite Robustness Index (CRI) as a weighted sum of individual stability dimensions.
    """
    cri = (
        weights.get("prediction_stability", 0.3) * prediction_stability +
        weights.get("calibration_stability", 0.2) * calibration_stability +
        weights.get("explanation_stability", 0.2) * explanation_stability +
        weights.get("representation_stability", 0.1) * representation_stability +
        weights.get("ood_performance", 0.2) * ood_performance
    )
    return float(cri)

def fit_noise_response_curve(
    noise_levels: List[float],
    metric_values: List[float]
) -> Dict[str, float]:
    """
    Fits degradation response curves over continuous noise levels.
    Reports:
    - Area under the degradation curve (AUDC)
    - Degradation slope
    - Half-performance threshold (noise level where metric drops by 50%)
    """
    x = np.array(noise_levels)
    y = np.array(metric_values)
    
    # 1. Area Under Curve (AUDC) via trapezoid rule
    norm_x = (x - x[0]) / (x[-1] - x[0] + 1e-12)
    audc = float(np.trapz(y, norm_x))
    
    # 2. Linear slope
    slope, _, _, _, _ = stats.linregress(x, y)
    
    # 3. Half-performance threshold
    half_y = y[0] * 0.5
    threshold = float(x[-1])  # default fallback
    for i in range(1, len(x)):
        if y[i] <= half_y:
            # Linear interpolation
            x_prev, x_curr = x[i-1], x[i]
            y_prev, y_curr = y[i-1], y[i]
            if abs(y_curr - y_prev) > 1e-12:
                threshold = float(x_prev + (half_y - y_prev) * (x_curr - x_prev) / (y_curr - y_prev))
            break
            
    return {
        "audc": audc,
        "slope": float(slope) if not np.isnan(slope) else 0.0,
        "half_performance_threshold": threshold
    }

def compute_explanation_drift(
    orig_attributions: np.ndarray,
    pert_attributions: np.ndarray
) -> Dict[str, float]:
    """
    Evaluates explanation drift using:
    - Cosine similarity of attribution vectors
    - Spearman rank correlation
    """
    # 1. Cosine similarity
    denom = (np.linalg.norm(orig_attributions) * np.linalg.norm(pert_attributions))
    if denom == 0:
        cosine = 1.0
    else:
        cosine = float(np.dot(orig_attributions, pert_attributions) / denom)
        
    # 2. Spearman rank
    spearman, _ = stats.spearmanr(orig_attributions, pert_attributions)
    if np.isnan(spearman):
        spearman = 0.0
        
    return {
        "cosine_similarity": cosine,
        "spearman_rho": float(spearman)
    }
