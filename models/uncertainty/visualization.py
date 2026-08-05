import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from typing import Dict, List, Any
from pathlib import Path

def plot_reliability_diagram(
    nominal_levels: List[float],
    empirical_coverages: List[float],
    save_path: str
) -> None:
    """
    Plots the regression reliability diagram showing nominal vs empirical coverage.
    """
    plt.figure(figsize=(6, 6))
    plt.plot(nominal_levels, empirical_coverages, 'o-', color='#1f77b4', label='Empirical Coverage', linewidth=2)
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')
    plt.xlabel('Nominal Confidence Level', fontsize=12)
    plt.ylabel('Empirical Coverage Rate', fontsize=12)
    plt.title('Regression Reliability Diagram', fontsize=14, fontweight='bold')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_prediction_intervals(
    targets: np.ndarray,
    predictions: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    save_path: str,
    max_samples: int = 50
) -> None:
    """
    Visualizes the true targets compared to predicted means and 95% prediction intervals.
    """
    n_samples = len(targets)
    display_samples = min(n_samples, max_samples)
    
    # Sort by target value for clear visual progression
    indices = np.argsort(targets)[:display_samples]
    
    t_disp = targets[indices]
    p_disp = predictions[indices]
    l_disp = lower_bounds[indices]
    u_disp = upper_bounds[indices]
    
    plt.figure(figsize=(10, 5))
    x = np.arange(display_samples)
    
    # Draw interval bars
    plt.vlines(x, l_disp, u_disp, colors='#aec7e8', label='95% Prediction Interval', alpha=0.8, linewidth=2)
    
    # Draw points
    plt.scatter(x, t_disp, color='#d62728', label='True Affinity', marker='x', s=40, zorder=5)
    plt.scatter(x, p_disp, color='#1f77b4', label='Predicted Affinity (Mean)', marker='o', s=30, zorder=4)
    
    plt.xlabel('Complex Index (Sorted by Target Value)', fontsize=12)
    plt.ylabel('Binding Affinity ($pK_d/pK_i$)', fontsize=12)
    plt.title('Prediction Intervals & Affinity Estimation', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_risk_coverage_curve(
    coverages: List[float],
    rmses: List[float],
    save_path: str
) -> None:
    """
    Plots the selective prediction risk-coverage curve (RMSE vs remaining samples fraction).
    """
    plt.figure(figsize=(7, 5))
    plt.plot(coverages, rmses, 's-', color='#2ca02c', linewidth=2, label='Selective Model')
    # Horizontal line representing base RMSE
    plt.axhline(y=rmses[0], color='red', linestyle='--', label='Unfiltered Baseline')
    plt.xlabel('Coverage Fraction (Remaining Samples)', fontsize=12)
    plt.ylabel('Validation RMSE', fontsize=12)
    plt.title('Risk-Coverage Trade-off Curve', fontsize=14, fontweight='bold')
    plt.xlim([min(coverages) - 0.05, 1.05])
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_uncertainty_distributions(
    aleatoric: np.ndarray,
    epistemic: np.ndarray,
    total: np.ndarray,
    save_path: str
) -> None:
    """
    Plots the histograms comparing Aleatoric, Epistemic, and Total uncertainty.
    """
    plt.figure(figsize=(8, 5))
    
    # Use logarithmic or standard bins based on variance magnitude
    bins = np.linspace(0, max(np.max(total), 0.1), 30)
    
    plt.hist(total, bins=bins, alpha=0.5, color='#1f77b4', label='Total Variance')
    plt.hist(epistemic, bins=bins, alpha=0.5, color='#ff7f0e', label='Epistemic (Model)')
    plt.hist(aleatoric, bins=bins, alpha=0.5, color='#2ca02c', label='Aleatoric (Data)')
    
    plt.xlabel('Predictive Variance (Uncertainty)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Uncertainty Decomposition Spectrum', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_confidence_vs_error(
    confidence: np.ndarray,
    abs_errors: np.ndarray,
    save_path: str
) -> None:
    """
    Scatter plot showing error vs confidence, with running average.
    """
    plt.figure(figsize=(7, 5))
    plt.scatter(confidence, abs_errors, alpha=0.4, color='#9467bd', label='Samples')
    
    # Fit running trend line (using linear regression of error vs confidence)
    if len(confidence) > 1:
        slope, intercept, r_val, p_val, std_err = stats.linregress(confidence, abs_errors)
        x_trend = np.linspace(min(confidence), max(confidence), 100)
        y_trend = slope * x_trend + intercept
        plt.plot(x_trend, y_trend, color='darkred', linewidth=2.5, 
                 label=f'Trend ($R = {r_val:.2f}$)')
                 
    plt.xlabel('Calibrated Confidence Score', fontsize=12)
    plt.ylabel('Absolute Prediction Error ($pK_d/pK_i$)', fontsize=12)
    plt.title('Confidence-Error Correlation Diagram', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_ood_uncertainty_histograms(
    in_dist_uncertainties: np.ndarray,
    ood_uncertainties: np.ndarray,
    save_path: str
) -> None:
    """
    Plots overlapping histograms showing the distribution of uncertainty for
    In-Distribution (ID) vs Out-of-Distribution (OOD) samples.
    """
    plt.figure(figsize=(8, 5))
    max_val = max(np.max(in_dist_uncertainties), np.max(ood_uncertainties), 0.1)
    bins = np.linspace(0, max_val, 30)
    
    plt.hist(in_dist_uncertainties, bins=bins, alpha=0.5, color='#1f77b4', density=True, label='In-Distribution (ID)')
    plt.hist(ood_uncertainties, bins=bins, alpha=0.5, color='#d62728', density=True, label='Out-of-Distribution (OOD)')
    
    plt.xlabel('Predictive Uncertainty (Standard Deviation)', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.title('Uncertainty Distribution under Covariate Shift', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
