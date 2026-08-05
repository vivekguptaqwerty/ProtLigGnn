import torch
from typing import Tuple

def calibrate_variance(variance: torch.Tensor, multiplier: float) -> torch.Tensor:
    """
    Scales predictive variance by a multiplier factor (variance scaling).
    """
    return variance * multiplier

def compute_confidence(variance: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Maps predictive variance (uncertainty) to a normalized confidence score in [0, 1].
    Higher variance maps to lower confidence using: confidence = exp(-variance / temperature).
    """
    # Clip variance to be non-negative
    variance = torch.clamp(variance, min=0.0)
    return torch.exp(-variance / max(temperature, 1e-6))

def compute_prediction_interval(
    mean: torch.Tensor, 
    variance: torch.Tensor, 
    confidence_level: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes empirical prediction intervals under assumption of normal distribution.
    For confidence_level=0.95, bounds = mean +/- 1.96 * std_dev.
    """
    std_dev = torch.sqrt(torch.clamp(variance, min=0.0))
    
    # Quantile critical value for normal distribution
    if abs(confidence_level - 0.95) < 1e-4:
        z = 1.95996
    elif abs(confidence_level - 0.90) < 1e-4:
        z = 1.64485
    elif abs(confidence_level - 0.99) < 1e-4:
        z = 2.57583
    else:
        # Fallback approximation for standard normal inverse CDF
        z = 1.96
        
    margin = z * std_dev
    return mean - margin, mean + margin
