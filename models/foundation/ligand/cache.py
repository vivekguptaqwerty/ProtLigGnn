import os
import json
import time
import hashlib
import torch
import rdkit
from pathlib import Path
from typing import Tuple

from models.foundation.ligand.config import LigandFoundationConfig
from models.foundation.base_encoder import FoundationModel

def get_smiles_hash(smiles: str) -> str:
    return hashlib.sha256(smiles.encode("utf-8")).hexdigest()

def get_tensor_checksum(tensor: torch.Tensor) -> float:
    return float(torch.sum(torch.abs(tensor)).item())

def load_ligand_with_cache(pdb_id: str, smiles: str, config: LigandFoundationConfig, encoder: FoundationModel) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """
    Loads molecular foundation embeddings from cache if valid, or extracts and updates the cache.
    
    Returns:
        atom_embs: Tensor of shape (L, D)
        mol_emb: Tensor of shape (D)
        is_emulator: Boolean indicating if emulator was used
    """
    cache_dir = Path(config.cache_directory)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    pt_path = cache_dir / f"{pdb_id}_embedding.pt"
    json_path = cache_dir / f"{pdb_id}_metadata.json"
    
    smiles_hash = get_smiles_hash(smiles)
    rdkit_ver = rdkit.__version__
    
    # 1. Attempt cache read
    if pt_path.exists() and json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                
            # Verify all metadata criteria
            valid = (
                meta.get("cache_schema_version") == config.cache_schema_version and
                meta.get("embedding_model") == config.ligand_model and
                meta.get("projection_dim") == config.projection_dimension and
                meta.get("smiles_hash") == smiles_hash and
                meta.get("rdkit_version") == rdkit_ver
            )
            
            if valid:
                data = torch.load(pt_path, map_location="cpu")
                atom_embs = data["atom_embeddings"]
                mol_emb = data["molecule_embedding"]
                is_emulator = data.get("is_emulator", False)
                
                # Checksum validation
                checksum_ok = (abs(get_tensor_checksum(atom_embs) - meta.get("checksum", 0.0)) < 1e-3)
                if checksum_ok:
                    return atom_embs, mol_emb, is_emulator
        except Exception as e:
            print(f"Warning: Failed to load cached ligand embedding for {pdb_id}: {e}. Regenerating...")
            
    # 2. Extract embedding
    res_dict = encoder.encode(smiles)
    atom_embs = res_dict["atom_embeddings"].cpu()
    mol_emb = res_dict["molecule_embedding"].cpu()
    is_emulator = res_dict["is_emulator"].item()
    
    # 3. Save cache
    try:
        torch.save({
            "atom_embeddings": atom_embs,
            "molecule_embedding": mol_emb,
            "is_emulator": is_emulator
        }, pt_path)
        
        meta = {
            "cache_schema_version": config.cache_schema_version,
            "embedding_model": config.ligand_model,
            "embedding_revision": encoder.model_id if not is_emulator else "emulator",
            "projection_dim": config.projection_dimension,
            "smiles_hash": smiles_hash,
            "canonicalization_method": "rdkit_canonical",
            "rdkit_version": rdkit_ver,
            "created_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "checksum": get_tensor_checksum(atom_embs)
        }
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4)
    except Exception as e:
        print(f"Warning: Failed to save cached ligand embedding for {pdb_id}: {e}")
        
    return atom_embs, mol_emb, is_emulator
