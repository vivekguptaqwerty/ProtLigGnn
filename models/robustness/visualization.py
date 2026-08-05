import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any

def plot_degradation_curves(
    noise_levels: List[float],
    rmse_values: List[float],
    ece_values: List[float],
    save_path: str
) -> None:
    """
    Plots prediction degradation and calibration drift curves side-by-side over noise sweeps.
    """
    plt.figure(figsize=(9, 4))
    
    # RMSE vs. Noise
    plt.subplot(1, 2, 1)
    plt.plot(noise_levels, rmse_values, 'r-o', linewidth=2)
    plt.xlabel('Perturbation Sigma (Å)', fontsize=10)
    plt.ylabel('RMSE (Å)', fontsize=10)
    plt.title('Prediction Sensitivity Curve', fontsize=11)
    plt.grid(True, linestyle='--')

    # ECE vs. Noise
    plt.subplot(1, 2, 2)
    plt.plot(noise_levels, ece_values, 'b-s', linewidth=2)
    plt.xlabel('Perturbation Sigma (Å)', fontsize=10)
    plt.ylabel('Expected Calibration Error (ECE)', fontsize=10)
    plt.title('Calibration Stability Curve', fontsize=11)
    plt.grid(True, linestyle='--')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_ood_uncertainty_comparison(
    in_dist_vars: np.ndarray,
    ood_vars: np.ndarray,
    save_path: str
) -> None:
    """
    Plots histograms of predictive uncertainty under in-distribution vs. OOD conditions.
    """
    plt.figure(figsize=(6, 4))
    plt.hist(in_dist_vars, bins=15, alpha=0.6, label='In-Distribution (ID)', color='green')
    plt.hist(ood_vars, bins=15, alpha=0.6, label='Out-of-Distribution (OOD)', color='red')
    plt.xlabel('Predictive Variance (Uncertainty)', fontsize=11)
    plt.ylabel('Count', fontsize=11)
    plt.title('Predictive Uncertainty under Distribution Shift', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
