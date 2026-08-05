import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_coordinate_drift_distribution(
    drifts: np.ndarray, 
    save_path: Path
) -> None:
    """
    Plots a histogram distribution of coordinate shifts after EGNN updates.
    """
    plt.figure(figsize=(8, 6), dpi=300)
    plt.hist(drifts, bins=30, color="#4C72B0", edgecolor="black", alpha=0.7)
    plt.axvline(np.mean(drifts), color="red", linestyle="dashed", linewidth=1.5, label=f"Mean: {np.mean(drifts):.4f} Å")
    plt.title("Coordinate Update Drift Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Coordinate Drift Magnitude (Å)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(save_path), bbox_inches="tight")
    plt.close()


def plot_pairwise_distance_preservation(
    dist_in: np.ndarray, 
    dist_out: np.ndarray, 
    save_path: Path
) -> None:
    """
    Plots a scatter plot comparing pairwise distances before vs. after message updates.
    """
    plt.figure(figsize=(7, 7), dpi=300)
    plt.scatter(dist_in.flatten(), dist_out.flatten(), alpha=0.3, color="#55A868", s=2)
    # Perfect diagonal reference line
    lims = [
        np.min([plt.xlim(), plt.ylim()]),
        np.max([plt.xlim(), plt.ylim()])
    ]
    plt.plot(lims, lims, "r--", alpha=0.75, zorder=0)
    plt.title("Pairwise Distance Preservation", fontsize=14, fontweight="bold")
    plt.xlabel("Original Distances (Å)", fontsize=12)
    plt.ylabel("Updated Distances (Å)", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(save_path), bbox_inches="tight")
    plt.close()
