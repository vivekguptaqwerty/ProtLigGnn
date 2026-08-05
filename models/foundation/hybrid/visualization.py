import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def plot_cross_attention_heatmap(
    attn_matrix: np.ndarray, 
    row_labels: list, 
    col_labels: list, 
    save_path: Path,
    title: str = "Cross-Attention Heatmap"
) -> None:
    """
    Plots a high-resolution heatmap for cross-graph attention weights.
    """
    plt.figure(figsize=(10, 8), dpi=300)
    sns.heatmap(
        attn_matrix, 
        xticklabels=col_labels, 
        yticklabels=row_labels, 
        cmap="viridis", 
        cbar=True,
        linewidths=0.5
    )
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Key Tokens / Atoms", fontsize=12)
    plt.ylabel("Query Tokens / Residues", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(save_path), bbox_inches="tight")
    plt.close()


def plot_modality_contributions(
    protein_contrib: float,
    ligand_contrib: float,
    save_path: Path
) -> None:
    """
    Plots a donut chart indicating the percentage contribution of protein vs. ligand features.
    """
    plt.figure(figsize=(6, 6), dpi=300)
    labels = ["Protein modality", "Ligand modality"]
    sizes = [protein_contrib, ligand_contrib]
    colors = ["#4C72B0", "#DD8452"]
    
    plt.pie(
        sizes, 
        labels=labels, 
        colors=colors, 
        autopct="%1.1f%%", 
        startangle=90, 
        pctdistance=0.85,
        textprops={"fontsize": 12, "weight": "bold"}
    )
    
    # Donut center circle
    centre_circle = plt.Circle((0,0), 0.70, fc="white")
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    
    plt.title("Modality Variance Contributions", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(str(save_path), bbox_inches="tight")
    plt.close()
