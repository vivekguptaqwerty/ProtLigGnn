import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from rdkit import Chem
from rdkit.Chem import rdPartialCharges

class BiophysicalFeatureExtractor:
    """
    Computes biophysical and solvation properties for protein pocket nodes and ligand heavy atoms.
    Properties include Gasteiger partial atomic charges, SASA approximations, and H-bond directional vectors.
    """
    def __init__(self):
        pass

    @staticmethod
    def compute_gasteiger_charges(rdkit_mol: Chem.Mol) -> np.ndarray:
        """Computes Gasteiger partial charges for each heavy atom in an RDKit Mol object."""
        if rdkit_mol is None:
            return np.zeros(1, dtype=np.float32)
            
        mol_copy = Chem.Mol(rdkit_mol)
        try:
            rdPartialCharges.ComputeGasteigerCharges(mol_copy)
            charges = []
            for atom in mol_copy.GetAtoms():
                if atom.GetAtomicNum() > 1: # Heavy atoms only
                    try:
                        val = float(atom.GetProp('_GasteigerCharge'))
                        charges.append(val if not np.isnan(val) and not np.isinf(val) else 0.0)
                    except Exception:
                        charges.append(0.0)
            return np.array(charges, dtype=np.float32)
        except Exception:
            num_heavy = sum(1 for a in rdkit_mol.GetAtoms() if a.GetAtomicNum() > 1)
            return np.zeros(max(num_heavy, 1), dtype=np.float32)

    @staticmethod
    def compute_sasa_estimate(coords: np.ndarray, radii: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Estimates Solvent Accessible Surface Area (SASA) using a fast pairwise sphere occlusion model.
        coords: [N, 3] spatial positions.
        """
        N = coords.shape[0]
        if N == 0:
            return np.zeros(0, dtype=np.float32)
            
        if radii is None:
            radii = np.full(N, 1.8, dtype=np.float32) # Default atomic radius ~1.8 Angstroms
            
        probe_radius = 1.4 # Water probe radius in Angstroms
        r_eff = radii + probe_radius
        sasa = 4.0 * np.pi * (r_eff ** 2)
        
        # Pairwise distance occlusion penalty
        dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        np.fill_diagonal(dist_matrix, np.inf)
        
        occlusion_counts = np.sum(dist_matrix < 4.5, axis=-1)
        sasa_scaled = sasa * np.exp(-0.15 * occlusion_counts)
        return sasa_scaled.astype(np.float32)

    @staticmethod
    def extract_hbond_vectors(rdkit_mol: Chem.Mol, coords: np.ndarray) -> np.ndarray:
        """
        Extracts 6-dimensional H-bond donor/acceptor indicator and directional vectors for each heavy atom.
        Returns array of shape [N, 6].
        """
        N = coords.shape[0]
        features = np.zeros((N, 6), dtype=np.float32)
        
        if rdkit_mol is None:
            return features
            
        heavy_idx = 0
        for atom in rdkit_mol.GetAtoms():
            if atom.GetAtomicNum() == 1:
                continue
                
            if heavy_idx >= N:
                break
                
            symbol = atom.GetSymbol()
            num_h = atom.GetTotalNumHs()
            
            # H-Bond Donor check (N/O with attached hydrogens)
            is_donor = 1.0 if symbol in ['N', 'O'] and num_h > 0 else 0.0
            
            # H-Bond Acceptor check (N/O/F with lone pairs)
            is_acceptor = 1.0 if symbol in ['N', 'O', 'F'] else 0.0
            
            features[heavy_idx, 0] = is_donor
            features[heavy_idx, 1] = is_acceptor
            
            # Compute local unit vector direction relative to neighbors
            neighbors = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
            if len(neighbors) > 0 and heavy_idx < coords.shape[0]:
                vec = np.zeros(3, dtype=np.float32)
                curr_pos = coords[heavy_idx]
                for nbr in neighbors:
                    nbr_idx = nbr.GetIdx()
                    if nbr_idx < coords.shape[0]:
                        diff = curr_pos - coords[nbr_idx]
                        norm = np.linalg.norm(diff)
                        if norm > 1e-5:
                            vec += diff / norm
                norm_v = np.linalg.norm(vec)
                if norm_v > 1e-5:
                    vec /= norm_v
                features[heavy_idx, 2:5] = vec
                
            features[heavy_idx, 5] = float(len(neighbors))
            heavy_idx += 1
            
        return features

class BiophysicalFeatureEncoder(nn.Module):
    """PyTorch module to project biophysical vectors into neural hidden dimensions."""
    def __init__(self, in_dim: int = 8, out_dim: int = 256):
        super(BiophysicalFeatureEncoder, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.SiLU(),
            nn.Linear(64, out_dim),
            nn.LayerNorm(out_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
