import os
import re
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any

TO_SINGLE_LETTER = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "UNK": "X"
}

AA_TO_INDEX = {
    "ALA": 0, "ARG": 1, "ASN": 2, "ASP": 3, "CYS": 4,
    "GLN": 5, "GLU": 6, "GLY": 7, "HIS": 8, "ILE": 9,
    "LEU": 10, "LYS": 11, "MET": 12, "PHE": 13, "PRO": 14,
    "SER": 15, "THR": 16, "TRP": 17, "TYR": 18, "VAL": 19,
    "ASX": 2, "GLX": 6, "UNK": 7, "MSE": 12
}

def get_pocket_residues(protein_path: Path, ligand_positions: np.ndarray) -> List[Tuple[str, int, str]]:
    """
    Identifies protein residues near the ligand binding site (within 6.0 Å cutoff) in PDB file order.
    Exactly replicates the logic of build_protein_pocket_graph.
    """
    atom_coords = []
    atom_res_ids = []
    residues_data = {}
    
    with open(protein_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(("ATOM  ", "HETATM")):
                try:
                    resname = line[17:20].strip().upper()
                    if resname not in AA_TO_INDEX:
                        continue
                    chain_id = line[21]
                    res_seq = int(line[22:26].strip())
                    atom_name = line[12:16].strip().upper()
                    
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    
                    element = line[76:78].strip().upper()
                    if not element and len(atom_name) > 0:
                        element = atom_name[0]
                except ValueError:
                    continue
                
                res_id = (chain_id, res_seq, resname)
                coord = np.array([x, y, z], dtype=np.float32)
                
                if res_id not in residues_data:
                    residues_data[res_id] = {
                        "resname": resname,
                        "atom_names": set(),
                        "ca_coord": None,
                        "atoms": [],
                        "non_h_atoms": []
                    }
                
                residues_data[res_id]["atom_names"].add(atom_name)
                residues_data[res_id]["atoms"].append(coord)
                if element != "H":
                    residues_data[res_id]["non_h_atoms"].append(coord)
                if atom_name == "CA":
                    residues_data[res_id]["ca_coord"] = coord
                
                if element != "H":
                    atom_coords.append(coord)
                    atom_res_ids.append(res_id)
                    
    if not atom_coords:
        return []
        
    coords_arr = np.array(atom_coords, dtype=np.float32)
    diff = coords_arr[:, None, :] - ligand_positions[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    
    near_mask = np.any(dists <= 6.0, axis=-1)
    pocket_res_ids = set()
    for idx in np.where(near_mask)[0]:
        pocket_res_ids.add(atom_res_ids[idx])
        
    sorted_pocket_res_ids = [res_id for res_id in residues_data.keys() if res_id in pocket_res_ids]
    return sorted_pocket_res_ids


def extract_and_align_protein(protein_path: Path, pocket_res_ids: List[Tuple[str, int, str]]) -> Tuple[str, List[int], Dict[str, Any]]:
    """
    Extracts the full protein sequence from a PDB file and aligns the pocket residue nodes to it.
    
    Args:
        protein_path: Path to the pdb file.
        pocket_res_ids: List of (chain_id, res_seq, resname) tuples representing pocket residues in order.
        
    Returns:
        full_sequence: Str sequence of the full protein.
        mapping_indices: List of integer indices mapping pocket residues to the full sequence index.
        stats: Dict of alignment metrics for verification reporting.
    """
    full_pdb_residues = []
    residues_seen = set()
    
    # Parse residues in PDB file order
    with open(protein_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(("ATOM  ", "HETATM")):
                try:
                    resname = line[17:20].strip().upper()
                    if resname not in TO_SINGLE_LETTER:
                        continue
                    chain_id = line[21]
                    res_seq = int(line[22:26].strip())
                except ValueError:
                    continue
                
                res_key = (chain_id, res_seq, resname)
                if res_key not in residues_seen:
                    residues_seen.add(res_key)
                    full_pdb_residues.append(res_key)
                    
    # Construct sequence
    full_sequence = "".join([TO_SINGLE_LETTER.get(r[2], "X") for r in full_pdb_residues])
    
    mapping_indices = []
    matched_count = 0
    unmatched_count = 0
    duplicates_count = 0
    mapped_positions = set()
    
    for res_id in pocket_res_ids:
        # Find index in full pdb residues list
        try:
            idx = full_pdb_residues.index(res_id)
            mapping_indices.append(idx)
            matched_count += 1
            if idx in mapped_positions:
                duplicates_count += 1
            mapped_positions.add(idx)
        except ValueError:
            # Fallback to chain & sequence matches if resname differs
            found = False
            for idx, r in enumerate(full_pdb_residues):
                if r[0] == res_id[0] and r[1] == res_id[1]:
                    mapping_indices.append(idx)
                    matched_count += 1
                    if idx in mapped_positions:
                        duplicates_count += 1
                    mapped_positions.add(idx)
                    found = True
                    break
            if not found:
                mapping_indices.append(-1)
                unmatched_count += 1
                
    total_pocket = len(pocket_res_ids)
    accuracy = (matched_count - duplicates_count) / max(1, total_pocket)
    pocket_coverage = total_pocket / max(1, len(full_pdb_residues))
    failure_rate = unmatched_count / max(1, total_pocket)
    
    stats = {
        "total_protein_residues": len(full_pdb_residues),
        "total_pocket_residues": total_pocket,
        "matched_residues": matched_count,
        "unmatched_residues": unmatched_count,
        "duplicate_mappings": duplicates_count,
        "pocket_coverage": pocket_coverage,
        "alignment_accuracy": accuracy,
        "failure_rate": failure_rate
    }
    
    return full_sequence, mapping_indices, stats
