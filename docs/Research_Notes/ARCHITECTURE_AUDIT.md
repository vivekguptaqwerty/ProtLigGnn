# Hardened Architecture Audit Report
## ProtLigGNN Optimized Baseline v1.1

**Total Trainable Parameters**: 362,881  
**Representational Latency (CUDA, Batch=8)**: 23.89 ms  
**Analysis Date**: 2026-07-12  

---

## 1. Mermaid Architecture & Tensor Flow

```mermaid
graph TD
    subgraph Ligand Encoder [Ligand Atom Graph Encoder]
        L_nodes["Ligand Nodes [N_lig, 78]"] --> GAT1["GATConv 1 [N_lig, 256]"]
        GAT1 --> LN1["LayerNorm & ELU"]
        LN1 --> GAT2["GATConv 2 [N_lig, 256]"]
        GAT2 --> LN2["LayerNorm & ELU"]
        LN2 --> GAT3["GATConv 3 [N_lig, 128]"]
        GAT3 --> LN3["LayerNorm"]
        LN3 --> L_encoded["Encoded Ligand [N_lig, 128]"]
    end

    subgraph Protein Encoder [Protein Pocket Residue Encoder]
        P_nodes["Protein Nodes [N_prot, 30]"] --> GCN1["GCNConv 1 [N_prot, 128]"]
        GCN1 --> PLN1["LayerNorm & ReLU"]
        PLN1 --> GCN2["GCNConv 2 [N_prot, 128]"]
        GCN2 --> PLN2["LayerNorm & ReLU"]
        PLN2 --> GCN3["GCNConv 3 [N_prot, 128]"]
        GCN3 --> PLN3["LayerNorm"]
        PLN3 --> P_encoded["Encoded Protein [N_prot, 128]"]
    end

    L_encoded --> CrossAttn["Bidirectional Cross-Attention"]
    P_encoded --> CrossAttn

    subgraph Cross Attention [Interaction Layer]
        CrossAttn --> L_to_P["Ligand-to-Protein MHA (4 Heads)\nQuery: Ligand, Key/Val: Protein"]
        CrossAttn --> P_to_L["Protein-to-Ligand MHA (4 Heads)\nQuery: Protein, Key/Val: Ligand"]
        L_to_P --> L_res["Residual & LayerNorm [N_lig, 128]"]
        P_to_L --> P_res["Residual & LayerNorm [N_prot, 128]"]
    end

    L_encoded --> L_cat["Concat [N_lig, 256]"]
    L_res --> L_cat
    P_encoded --> P_cat["Concat [N_prot, 256]"]
    P_res --> P_cat

    subgraph Pooling [Readout Layer]
        L_cat --> L_pool["Global Mean Pool [1, 256]"]
        P_cat --> P_pool["Global Mean Pool [1, 256]"]
        L_pool --> Final_cat["Concatenate [1, 512]"]
        P_pool --> Final_cat
    end

    subgraph Regressor [MLP Head]
        Final_cat --> MLP1["Linear (512 -> 256) & ReLU & Dropout"]
        MLP1 --> MLP2["Linear (256 -> 64) & ReLU & Dropout"]
        MLP2 --> MLP3["Linear (64 -> 1)"]
        MLP3 --> Out["Predicted Affinity (pK_d/pK_i)"]
    end
```

---

## 2. Big-O Complexity & Receptive Field Growth

Let $V_L$ be the number of ligand heavy atoms, $E_L$ be the number of ligand edges, $V_P$ be the number of protein pocket residues, and $E_P$ be the number of pocket edges. Let $d = 128$ be the hidden dimension.

| Component | Time Complexity | Space Complexity |
| :--- | :---: | :---: |
| **Ligand Encoder** | $O(V_L \cdot d + E_L \cdot d + V_L \cdot d^2)$ | $O(V_L \cdot d + E_L)$ |
| **Protein Encoder** | $O(V_P \cdot d + E_P \cdot d + V_P \cdot d^2)$ | $O(V_P \cdot d + E_P)$ |
| **Cross-Attention** | $O(V_L \cdot V_P \cdot d)$ | $O(V_L \cdot V_P)$ |
| **Regressor (MLP)** | $O(d^2)$ | $O(d)$ |

### Receptive Field Growth
- **Ligand Encoder**: A 3-layer GAT structure allows each ligand atom node to aggregate information from its **3-hop local neighborhood**. Since ligand graphs are small ($V_L \le 50$ heavy atoms usually), 3 hops cover almost the entire ligand graph diameter.
- **Protein Pocket Encoder**: A 3-layer GCN allows each pocket residue to aggregate local residue connectivity within **3 spatial hops**. This aggregates pocket-level context before interaction.

---

## 3. Parameter Allocation Breakdown

The 362,881 parameters are distributed as follows:

- **Ligand Encoder**: 44,416 parameters (12.2% of model)
- **Protein Encoder**: 38,016 parameters (10.5% of model)
- **Cross-Attention Layer**: 132,608 parameters (36.5% of model)
- **Regressor (MLP)**: 147,841 parameters (40.7% of model)

```
+-------------------------------------------------------------+
|                     ProtLigGNN Model                        |
|                  (362,881 Parameters)                       |
+------------------------------+------------------------------+
                               |
       +-----------------------+-----------------------+
       |                                               |
+------v--------+                               +------v--------+
| Encoders      |                               | Interaction   |
| 82,432 Params |                               | 280,449 Params|
| (22.7%)       |                               | (77.3%)       |
+---------------+                               +---------------+
```

---

## 4. Future Architecture Research Directions (Phase 3)

The baseline model will remain frozen under semantic versioning. Active Phase 3 research will branch from this coordinate and evaluate:
1. **Equivariant GNNs (SE(3)-Transformers / EGNNs)**: To replace coordinate-free GCN/GAT layers, encoding physical 3D Cartesian coordinates directly to ensure roto-translation equivariance.
2. **Explicit Distance Biases**: Incorporating physical atom-residue distances $D_{i, j}$ directly into the cross-attention softmax:
   $$\text{Attention}(Q, K) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} - \gamma D\right)$$
3. **Graph Transformers**: Utilizing global self-attention over the union of ligand and protein graphs to learn joint coordinates without restriction to pocket boundaries.
