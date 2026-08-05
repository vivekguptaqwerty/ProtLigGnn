import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'cm'

test_strings = [
    r"$f(\mathcal{P}, \mathcal{L}) \longrightarrow \mathrm{p}K_d = -\log_{10}(K_d / \mathrm{1 M})$",
    r"$e_{uv}^{(k)} = \exp\left(-\gamma (d(u,v) - \mu_k)^2\right), \quad k \in \{1, \dots, 32\}$",
    r"$\mathbf{A}_{ij}^{\mathrm{inter}} = \mathbb{I}(\|\mathbf{x}_i^P - \mathbf{x}_j^L\|_2 \le 12.0\text{ A})$".replace(r"\le", r"\leq"),
    r"$m_{ij} = \phi_m\left(h_i^l, h_j^l, \|\mathbf{x}_i^l - \mathbf{x}_j^l\|^2, e_{ij}\right)$",
    r"$\mathbf{x}_i^{l+1} = \mathbf{x}_i^l + \sum_{j \in \mathcal{N}(i)} (\mathbf{x}_i^l - \mathbf{x}_j^l) \cdot \phi_x(m_{ij})$",
    r"$h_i^{l+1} = \phi_h\left(h_i^l, \sum_{j \in \mathcal{N}(i)} m_{ij}\right)$",
    r"$\mathbf{S}_{ij} = \frac{\mathbf{Q}_i \mathbf{K}_j^\top}{\sqrt{d_k}} + \mathbf{B}_{ij}^{\mathrm{RBF}}, \quad \alpha_{ij} = \frac{\exp(\mathbf{S}_{ij})}{\sum_k \exp(\mathbf{S}_{ik})}$",
    r"$\mathbf{z}_{\mathrm{pocket}} = \sum_{i} \alpha_i \mathbf{V}_i^P, \quad \mathbf{z}_{\mathrm{ligand}} = \sum_{j} \beta_j \mathbf{V}_j^L$",
    r"$\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{NLL}}^{\mathrm{NIG}}(y, \gamma, v, \alpha, \beta) + \lambda_1 \mathcal{L}_{\mathrm{reg}} + \lambda_2 \mathcal{L}_{\mathrm{BCE}}(\hat{\mathbf{C}}, \mathbf{C})$",
    r"$\mathrm{Var}[y] = \frac{\beta}{v(\alpha - 1)}, \quad \mathrm{Epistemic\ Uncertainty} = \frac{\beta}{\alpha - 1}$"
]

fig, ax = plt.subplots(figsize=(6, 6))
for idx, s in enumerate(test_strings):
    try:
        ax.text(0.1, 0.9 - idx*0.08, s, fontsize=10)
        print(f"String {idx} OK")
    except Exception as e:
        print(f"String {idx} FAILED: {e}")

fig.savefig("outputs/pptx_assets/test_math.png")
print("Math test completed successfully!")
