import numpy as np
import scipy.stats as stats
from typing import Dict, List, Any, Tuple

def compute_prediction_metrics(
    predictions: np.ndarray,
    targets: np.ndarray
) -> Dict[str, float]:
    """
    Computes standard prediction performance metrics: RMSE, MAE, Pearson, Spearman, R2, Bias.
    """
    errors = predictions - targets
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(errors))
    bias = np.mean(errors)
    
    # Pearson Correlation Coefficient
    if len(np.unique(predictions)) > 1 and len(np.unique(targets)) > 1:
        pcc, _ = stats.pearsonr(predictions, targets)
        spearman, _ = stats.spearmanr(predictions, targets)
    else:
        pcc, spearman = 0.0, 0.0
        
    # R2 Score
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-10))
    
    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "pearson": float(pcc),
        "spearman": float(spearman),
        "r2": float(r2),
        "bias": float(bias),
        "mse": float(mse)
    }

def compute_bootstrap_ci(
    predictions: np.ndarray,
    targets: np.ndarray,
    n_iterations: int = 1000,
    confidence_level: float = 0.95
) -> Dict[str, Tuple[float, float]]:
    """
    Computes bootstrap confidence intervals for RMSE and MAE.
    """
    rmse_samples = []
    mae_samples = []
    n_samples = len(predictions)
    
    np.random.seed(42)
    for _ in range(n_iterations):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        pred_sample = predictions[indices]
        targ_sample = targets[indices]
        
        rmse_samples.append(np.sqrt(np.mean((pred_sample - targ_sample) ** 2)))
        mae_samples.append(np.mean(np.abs(pred_sample - targ_sample)))
        
    alpha = (1.0 - confidence_level) / 2.0
    lower_pct = alpha * 100
    upper_pct = (1.0 - alpha) * 100
    
    rmse_ci = (float(np.percentile(rmse_samples, lower_pct)), float(np.percentile(rmse_samples, upper_pct)))
    mae_ci = (float(np.percentile(mae_samples, lower_pct)), float(np.percentile(mae_samples, upper_pct)))
    
    return {
        "rmse_ci": rmse_ci,
        "mae_ci": mae_ci
    }
