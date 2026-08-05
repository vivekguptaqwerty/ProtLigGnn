import numpy as np
import scipy.stats as stats
from typing import Dict, List, Any, Tuple

def compute_regression_ece(
    predictions: np.ndarray,
    variances: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, float, List[float], List[float]]:
    """
    Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE) for regression.
    Measures the alignment between nominal confidence levels and empirical coverage rates.
    """
    nominal_levels = np.linspace(0.1, 0.9, n_bins)
    empirical_coverages = []
    
    std_devs = np.sqrt(np.maximum(variances, 1e-8))
    
    for p in nominal_levels:
        # Two-tailed normal quantile z-score
        z = stats.norm.ppf((1.0 + p) / 2.0)
        
        lower_bound = predictions - z * std_devs
        upper_bound = predictions + z * std_devs
        
        # Coverage fraction
        covered = (targets >= lower_bound) & (targets <= upper_bound)
        empirical_coverages.append(float(np.mean(covered)))
        
    errors = np.abs(np.array(empirical_coverages) - nominal_levels)
    ece = float(np.mean(errors))
    mce = float(np.max(errors))
    
    return ece, mce, list(nominal_levels), empirical_coverages

def compute_adaptive_ece(
    predictions: np.ndarray,
    variances: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Computes Adaptive ECE (AECE) by binning uncertainty values such that each bin 
    has an equal number of samples.
    """
    n_samples = len(predictions)
    if n_samples < n_bins:
        return 0.0
        
    # Sort samples by variance
    sorted_indices = np.argsort(variances)
    sorted_preds = predictions[sorted_indices]
    sorted_vars = variances[sorted_indices]
    sorted_targets = targets[sorted_indices]
    
    bin_size = n_samples // n_bins
    errors = []
    
    for b in range(n_bins):
        start_idx = b * bin_size
        end_idx = n_samples if b == n_bins - 1 else (b + 1) * bin_size
        
        b_preds = sorted_preds[start_idx:end_idx]
        b_vars = sorted_vars[start_idx:end_idx]
        b_targets = sorted_targets[start_idx:end_idx]
        b_stds = np.sqrt(np.maximum(b_vars, 1e-8))
        
        # We check ECE at the 95% nominal level for each bin
        z = 1.95996
        covered = (b_targets >= b_preds - z * b_stds) & (b_targets <= b_preds + z * b_stds)
        coverage = np.mean(covered)
        
        weight = (end_idx - start_idx) / n_samples
        errors.append(weight * abs(0.95 - coverage))
        
    return float(np.sum(errors))

def compute_ence(
    predictions: np.ndarray,
    variances: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Computes Expected Normalized Calibration Error (ENCE).
    Measures ECE in terms of expected vs actual error magnitude (RMSE vs Mean Std).
    """
    n_samples = len(predictions)
    if n_samples < n_bins:
        return 0.0
        
    # Sort by variance
    sorted_indices = np.argsort(variances)
    sorted_preds = predictions[sorted_indices]
    sorted_vars = sorted_vars_orig = variances[sorted_indices]
    sorted_targets = targets[sorted_indices]
    
    bin_size = n_samples // n_bins
    ence_accum = 0.0
    
    for b in range(n_bins):
        start_idx = b * bin_size
        end_idx = n_samples if b == n_bins - 1 else (b + 1) * bin_size
        
        b_preds = sorted_preds[start_idx:end_idx]
        b_vars = sorted_vars[start_idx:end_idx]
        b_targets = sorted_targets[start_idx:end_idx]
        
        rmse = np.sqrt(np.mean((b_preds - b_targets) ** 2))
        mean_std = np.mean(np.sqrt(np.maximum(b_vars, 1e-8)))
        
        if mean_std > 0:
            ence_val = abs(rmse - mean_std) / mean_std
            weight = (end_idx - start_idx) / n_samples
            ence_accum += weight * ence_val
            
    return float(ence_accum)

def compute_reliability_metrics(
    predictions: np.ndarray,
    variances: np.ndarray,
    targets: np.ndarray,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """
    Computes comprehensive reliability metrics: NLL, PICP, MPIW, Sharpness, Coverage Error, Correlations.
    """
    errors = predictions - targets
    abs_errors = np.abs(errors)
    std_devs = np.sqrt(np.maximum(variances, 1e-8))
    
    # 1. Negative Log-Likelihood (Gaussian assumption)
    nll = 0.5 * np.log(2.0 * np.pi * variances) + (errors ** 2) / (2.0 * variances)
    mean_nll = np.mean(nll)
    
    # 2. Prediction Interval Coverage Probability (PICP) at confidence_level (default 95%)
    z = stats.norm.ppf((1.0 + confidence_level) / 2.0)
    lower_bounds = predictions - z * std_devs
    upper_bounds = predictions + z * std_devs
    covered = (targets >= lower_bounds) & (targets <= upper_bounds)
    picp = np.mean(covered)
    
    # 3. Mean Prediction Interval Width (MPIW)
    widths = upper_bounds - lower_bounds
    mpiw = np.mean(widths)
    
    # 4. Sharpness (variance of widths)
    sharpness = np.std(widths)
    
    # 5. Coverage Error
    coverage_error = abs(confidence_level - picp)
    
    # 6. Correlations
    if len(np.unique(std_devs)) > 1 and len(np.unique(abs_errors)) > 1:
        uncertainty_error_corr, _ = stats.pearsonr(std_devs, abs_errors)
        spearman_corr, _ = stats.spearmanr(std_devs, abs_errors)
    else:
        uncertainty_error_corr, spearman_corr = 0.0, 0.0
        
    # ECE and MCE
    ece, mce, _, _ = compute_regression_ece(predictions, variances, targets)
    aece = compute_adaptive_ece(predictions, variances, targets)
    ence = compute_ence(predictions, variances, targets)
    
    return {
        "nll": float(mean_nll),
        "picp": float(picp),
        "mpiw": float(mpiw),
        "sharpness": float(sharpness),
        "coverage_error": float(coverage_error),
        "uncertainty_error_corr": float(uncertainty_error_corr),
        "uncertainty_error_spearman": float(spearman_corr),
        "ece": ece,
        "mce": mce,
        "aece": aece,
        "ence": ence
    }

def evaluate_selective_prediction(
    predictions: np.ndarray,
    variances: np.ndarray,
    targets: np.ndarray,
    rejection_fractions: List[float] = [0.05, 0.10, 0.20]
) -> List[Dict[str, Any]]:
    """
    Performs selective prediction trade-off analysis by rejecting samples with the highest uncertainty.
    """
    n_samples = len(predictions)
    sorted_indices = np.argsort(variances)  # sorted ascending (low variance first)
    
    base_rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    
    results = []
    for frac in rejection_fractions:
        keep_count = int(n_samples * (1.0 - frac))
        if keep_count == 0:
            continue
            
        keep_indices = sorted_indices[:keep_count]
        keep_preds = predictions[keep_indices]
        keep_targets = targets[keep_indices]
        
        filtered_rmse = np.sqrt(np.mean((keep_preds - keep_targets) ** 2))
        improvement = (base_rmse - filtered_rmse) / (base_rmse + 1e-10) * 100.0
        
        results.append({
            "rejection_fraction": frac,
            "coverage": 1.0 - frac,
            "remaining_samples": keep_count,
            "rmse": float(filtered_rmse),
            "relative_improvement_pct": float(improvement)
        })
        
    return results
