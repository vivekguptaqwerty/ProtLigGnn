# Pairwise Feature Specification (Refined)
**Status**: Proposed  
**Version**: 2.0  

This document details the refined features of the Chemistry Branch and the AtomicPropertyEncoder.

---

## 1. Reusable AtomicPropertyEncoder

The `AtomicPropertyEncoder` dynamically extracts physical and chemical properties of elements using RDKit Periodic Table lookups. It maps atomic numbers to:

| Property | Symbol / Data Type | source |
| :--- | :---: | :--- |
| **Atomic Number** | $Z$ (int) | `atom.GetAtomicNum()` |
| **Atomic Mass** | $M$ (float) | `atom.GetMass()` |
| **Electronegativity** | $\chi$ (float) | Pauling mapping |
| **Covalent Radius** | $R_{\text{cov}}$ (float) | `GetRcovalent()` |
| **vdW Radius** | $R_{\text{vdW}}$ (float) | `GetRvdw()` |
| **Group / Period** | $G, P$ (int) | Periodic Table metadata |
| **Valence Electrons** | $V$ (int) | `atom.GetNValence()` |
| **Common Valence** | $C$ (int) | Periodic Table metadata |
| **Metal Flag** | bool | Metal elements set membership |
| **Halogen Flag** | bool | Halogen set membership |
| **Noble Gas Flag** | bool | Noble gas set membership |

---

## 2. Chemistry Branch Features

For a ligand atom $i$ and protein residue $j$, we construct a multi-dimensional representation:

### 2.1 Ligand Atom Features ($\mathbf{x}^{raw}_{lig, i}$)
Dimension: 78. Features extracted directly from `ligand_batch.x`, plus the `AtomicPropertyEncoder` lookup.

### 2.2 Protein Residue Features ($\mathbf{x}^{raw}_{prot, j}$)
Dimension: 30. Features extracted directly from `protein_batch.x`.

### 2.3 Electronegativity Difference
Calculated as:
$$\Delta \chi_{ij} = |\chi_i - \chi_{\text{protein}}|$$
where $\chi_{\text{protein}} = 3.04$ (Nitrogen amide center) is used as the default reference for protein amide links.
