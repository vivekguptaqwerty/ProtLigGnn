import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Tuple
from pathlib import Path

def plot_faithfulness_curves(
    deletion_vals: List[float],
    insertion_vals: List[float],
    save_path: str
) -> None:
    """
    Plots the deletion (comprehensiveness) and insertion (sufficiency) curves side by side.
    """
    plt.figure(figsize=(8, 4))
    
    # Deletion Curve
    plt.subplot(1, 2, 1)
    plt.plot(np.linspace(0, 100, len(deletion_vals)), deletion_vals, 'r-o', label='Deletion (Comprehensiveness)')
    plt.xlabel('% Nodes Removed', fontsize=10)
    plt.ylabel('Predicted Affinity', fontsize=10)
    plt.title('Deletion Curve (AUC ↓)', fontsize=11)
    plt.grid(True, linestyle='--')
    plt.legend()

    # Insertion Curve
    plt.subplot(1, 2, 2)
    plt.plot(np.linspace(0, 100, len(insertion_vals)), insertion_vals, 'g-o', label='Insertion (Sufficiency)')
    plt.xlabel('% Nodes Inserted', fontsize=10)
    plt.ylabel('Predicted Affinity', fontsize=10)
    plt.title('Insertion Curve (AUC ↑)', fontsize=11)
    plt.grid(True, linestyle='--')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_explanation_agreement(
    agreement_matrix: np.ndarray,
    method_names: List[str],
    save_path: str
) -> None:
    """
    Plots a heatmap showing cross-method agreement (e.g. Spearman rank correlation).
    """
    plt.figure(figsize=(6, 5))
    im = plt.imshow(agreement_matrix, cmap='viridis', vmin=-1.0, vmax=1.0)
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Spearman Rho', fontsize=10)
    
    # Add text labels on cells
    for i in range(len(method_names)):
        for j in range(len(method_names)):
            plt.text(j, i, f"{agreement_matrix[i, j]:.2f}", 
                     ha="center", va="center", color="w" if abs(agreement_matrix[i, j]) < 0.5 else "black")
            
    plt.xticks(range(len(method_names)), method_names, rotation=45, ha='right', fontsize=9)
    plt.yticks(range(len(method_names)), method_names, fontsize=9)
    plt.title('Cross-Method Explanation Agreement', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_side_by_side_attributions(
    atom_attributions: Dict[str, np.ndarray],
    save_path: str
) -> None:
    """
    Plots side-by-side comparison of atom-level attributions for all four methods.
    """
    methods = list(atom_attributions.keys())
    num_methods = len(methods)
    
    plt.figure(figsize=(3 * num_methods, 3))
    
    for i, m in enumerate(methods):
        plt.subplot(1, num_methods, i + 1)
        vals = atom_attributions[m]
        plt.bar(range(len(vals)), vals, color='#1f77b4')
        plt.xlabel('Atom Index', fontsize=9)
        plt.ylabel('Score', fontsize=9)
        plt.title(m.replace('_', ' ').title(), fontsize=10)
        plt.ylim(0, 1.05)
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
