import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("outputs/pptx_assets/equations", exist_ok=True)
os.makedirs("outputs/pptx_assets", exist_ok=True)

# Set matplotlib style for high quality scientific math rendering
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'cm' # Computer Modern for publication LaTeX style

def create_equation_card(eq_text, filename, title="", width=6.5, height=1.8, bg_color='#0a192f', text_color='#00bfa5', title_color='#ffffff'):
    fig, ax = plt.subplots(figsize=(width, height), dpi=300)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.axis('off')
    
    if title:
        ax.text(0.05, 0.82, title, transform=ax.transAxes, color=title_color, 
                fontsize=11, fontweight='bold', ha='left', va='center')
        
    ax.text(0.5, 0.42 if title else 0.5, eq_text, transform=ax.transAxes, color=text_color, 
            fontsize=14, ha='center', va='center')
    
    # Add subtle border around math card
    rect = plt.Rectangle((0.01, 0.01), 0.98, 0.98, fill=False, edgecolor='#00bfa5', lw=1.2, transform=ax.transAxes, rx=10)
    ax.add_patch(rect)
    
    plt.tight_layout()
    fig.savefig(f"outputs/pptx_assets/equations/{filename}.png", bbox_inches='tight', facecolor=bg_color)
    plt.close(fig)

# 1. Slide 3 Equation
create_equation_card(
    r"$f(\mathcal{P}, \mathcal{L}) \longrightarrow \text{p}K_d = -\log_{10}(K_d)$",
    "eq_slide3",
    title="Thermodynamic Binding Affinity Mapping Function",
    width=6.0, height=1.4
)

# 2. Slide 9 RBF Equation
create_equation_card(
    r"$e_{uv}^{(k)} = \exp\left(-\gamma \left(d(u,v) - \mu_k\right)^2\right), \quad k \in \{1, \dots, 32\}$",
    "eq_slide9",
    title="Gaussian Radial Basis Function (RBF) Distance Encoding",
    width=6.5, height=1.5
)

# 3. Slide 13 Interaction Graph Equation
create_equation_card(
    r"$\mathbf{A}_{ij}^{\text{inter}} = \begin{cases} 1 & \text{if } \|\mathbf{x}_i^P - \mathbf{x}_j^L\|_2 \le 12.0\,\text{\AA} \\ 0 & \text{otherwise} \end{cases}$",
    "eq_slide13",
    title="Bipartite Pocket-Ligand Proximity Adjacency",
    width=6.5, height=1.6
)

# 4. Slide 14 EGNN Equations (Multi-line)
fig, ax = plt.subplots(figsize=(6.8, 2.2), dpi=300)
bg_color = '#0a192f'
fig.patch.set_facecolor(bg_color)
ax.set_facecolor(bg_color)
ax.axis('off')
ax.text(0.04, 0.88, "Equivariant Graph Neural Network (EGNN) SE(3) Layer Updates", transform=ax.transAxes, color='#ffffff', fontsize=11, fontweight='bold')
ax.text(0.5, 0.65, r"$m_{ij} = \phi_m\left(h_i^l, h_j^l, \|\mathbf{x}_i^l - \mathbf{x}_j^l\|^2, e_{ij}\right)$", transform=ax.transAxes, color='#00bfa5', fontsize=12, ha='center')
ax.text(0.5, 0.40, r"$\mathbf{x}_i^{l+1} = \mathbf{x}_i^l + \sum_{j \in \mathcal{N}(i)} (\mathbf{x}_i^l - \mathbf{x}_j^l) \cdot \phi_x(m_{ij})$", transform=ax.transAxes, color='#ffffff', fontsize=12, ha='center')
ax.text(0.5, 0.15, r"$h_i^{l+1} = \phi_h\left(h_i^l, \sum_{j \in \mathcal{N}(i)} m_{ij}\right)$", transform=ax.transAxes, color='#00bfa5', fontsize=12, ha='center')
rect = plt.Rectangle((0.01, 0.01), 0.98, 0.98, fill=False, edgecolor='#00bfa5', lw=1.2, transform=ax.transAxes)
ax.add_patch(rect)
plt.tight_layout()
fig.savefig("outputs/pptx_assets/equations/eq_slide14.png", bbox_inches='tight', facecolor=bg_color)
plt.close(fig)

# 5. Slide 15 Cross Attention Equations
fig, ax = plt.subplots(figsize=(6.8, 2.0), dpi=300)
fig.patch.set_facecolor(bg_color)
ax.set_facecolor(bg_color)
ax.axis('off')
ax.text(0.04, 0.86, "Distance-Biased Bidirectional Cross-Attention", transform=ax.transAxes, color='#ffffff', fontsize=11, fontweight='bold')
ax.text(0.5, 0.56, r"$\mathbf{S}_{ij} = \frac{\mathbf{Q}_i \mathbf{K}_j^\top}{\sqrt{d_k}} + \mathbf{B}_{ij}^{\text{RBF}}, \quad \alpha_{ij} = \frac{\exp(\mathbf{S}_{ij})}{\sum_k \exp(\mathbf{S}_{ik})}$", transform=ax.transAxes, color='#00bfa5', fontsize=12, ha='center')
ax.text(0.5, 0.22, r"$\mathbf{z}_{\text{pocket}} = \sum_{i} \alpha_i \mathbf{V}_i^P, \quad \mathbf{z}_{\text{ligand}} = \sum_{j} \beta_j \mathbf{V}_j^L$", transform=ax.transAxes, color='#ffffff', fontsize=12, ha='center')
rect = plt.Rectangle((0.01, 0.01), 0.98, 0.98, fill=False, edgecolor='#00bfa5', lw=1.2, transform=ax.transAxes)
ax.add_patch(rect)
plt.tight_layout()
fig.savefig("outputs/pptx_assets/equations/eq_slide15.png", bbox_inches='tight', facecolor=bg_color)
plt.close(fig)

# 6. Slide 16 Loss Formulation Equations
fig, ax = plt.subplots(figsize=(6.8, 2.0), dpi=300)
fig.patch.set_facecolor(bg_color)
ax.set_facecolor(bg_color)
ax.axis('off')
ax.text(0.04, 0.86, "Evidential NIG Regression & Contact Loss Objective", transform=ax.transAxes, color='#ffffff', fontsize=11, fontweight='bold')
ax.text(0.5, 0.56, r"$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{NLL}}^{\text{NIG}}(y, \gamma, v, \alpha, \beta) + \lambda_1 \mathcal{L}_{\text{reg}} + \lambda_2 \mathcal{L}_{\text{BCE}}(\hat{\mathbf{C}}, \mathbf{C})$", transform=ax.transAxes, color='#00bfa5', fontsize=11, ha='center')
ax.text(0.5, 0.22, r"$\text{Var}[y] = \frac{\beta}{v(\alpha - 1)}, \quad \text{Epistemic Uncertainty} = \frac{\beta}{\alpha - 1}$", transform=ax.transAxes, color='#ffffff', fontsize=11, ha='center')
rect = plt.Rectangle((0.01, 0.01), 0.98, 0.98, fill=False, edgecolor='#00bfa5', lw=1.2, transform=ax.transAxes)
ax.add_patch(rect)
plt.tight_layout()
fig.savefig("outputs/pptx_assets/equations/eq_slide16.png", bbox_inches='tight', facecolor=bg_color)
plt.close(fig)

print("Generated all equation cards successfully.")
