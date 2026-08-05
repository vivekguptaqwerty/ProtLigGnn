import os
import re
import json
import time
import hashlib
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
from models.foundation.base_encoder import FoundationModel
from models.foundation.protein.config import ProteinFoundationConfig

try:
    import transformers
    from transformers import AutoTokenizer, AutoModel, T5EncoderModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class DeterministicProteinEmulator:
    """
    Deterministic pseudo-embedding generator for offline test validation.
    Generates identical embeddings for the same sequence using SHA256 hashing seeds.
    """
    
    @staticmethod
    def generate(sequence: str, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = len(sequence)
        # Create a stable seed from sequence
        hasher = hashlib.sha256(sequence.encode("utf-8"))
        seed = int(hasher.hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        
        # Generate token level embeddings
        token_embs = rng.normal(0.0, 1.0, size=(seq_len, dim)).astype(np.float32)
        # Generate sequence level embedding
        seq_emb = np.mean(token_embs, axis=0)
        
        return torch.from_numpy(token_embs), torch.from_numpy(seq_emb)


class ESM2Encoder(FoundationModel):
    """
    Adapter for the ESM-2 protein language model (facebook/esm2_t6_8M_UR50D).
    """
    
    def __init__(self, config: ProteinFoundationConfig, device: str = "cpu") -> None:
        super().__init__(model_name="esm2", device=device)
        self.config = config
        self.model_id = "facebook/esm2_t6_8M_UR50D"
        self.embedding_dim = 320
        self.emulator_active = True
        
        # Check if we should try loading the real model
        use_real = (os.environ.get("LIGPROTGNN_PRODUCTION", "0") == "1")
        
        if TRANSFORMERS_AVAILABLE and use_real:
            try:
                print(f"Loading ESM-2 transformer ({self.model_id})...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self.model = AutoModel.from_pretrained(self.model_id)
                self.model = self.model.to(device)
                if config.freeze_backbone:
                    for param in self.model.parameters():
                        param.requires_grad = False
                    self.model.eval()
                self.emulator_active = False
            except Exception as e:
                print(f"Warning: Failed to load real ESM-2 model. Falling back to emulator: {e}")
                self.emulator_active = True
        else:
            self.emulator_active = True

    def encode(self, sequence: str) -> Dict[str, torch.Tensor]:
        if self.emulator_active:
            token_embs, seq_emb = DeterministicProteinEmulator.generate(sequence, self.embedding_dim)
            return {
                "residue_embeddings": token_embs.to(self.device),
                "sequence_embedding": seq_emb.to(self.device),
                "is_emulator": torch.tensor(True, device=self.device)
            }
            
        # Tokenize sequence
        inputs = self.tokenizer(sequence, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
        hidden_states = outputs.hidden_states
        
        # Layer Selection
        if self.config.layer_selection == "last4":
            # Mean of last 4 layers
            selected_embs = torch.stack(hidden_states[-4:], dim=0).mean(dim=0)[0]
        else:
            selected_embs = hidden_states[-1][0] # (seq_len_with_special_tokens, dim)
            
        # Remove special token representations (ESM uses <cls> at index 0 and <eos> at last index)
        seq_len = len(sequence)
        residue_embs = selected_embs[1:1 + seq_len]
        
        # Sequence level embedding
        seq_emb = residue_embs.mean(dim=0)
        
        return {
            "residue_embeddings": residue_embs,
            "sequence_embedding": seq_emb,
            "is_emulator": torch.tensor(False, device=self.device)
        }


class ProtT5Encoder(FoundationModel):
    """
    Adapter for the ProtT5 protein language model (Rostlab/prot_t5_xl_uniref50).
    """
    
    def __init__(self, config: ProteinFoundationConfig, device: str = "cpu") -> None:
        super().__init__(model_name="prot_t5", device=device)
        self.config = config
        self.model_id = "Rostlab/prot_t5_xl_uniref50"
        self.embedding_dim = 1024
        self.emulator_active = True
        
        use_real = (os.environ.get("LIGPROTGNN_PRODUCTION", "0") == "1")
        
        if TRANSFORMERS_AVAILABLE and use_real:
            try:
                print(f"Loading ProtT5 transformer ({self.model_id})...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                from transformers import T5EncoderModel
                self.model = T5EncoderModel.from_pretrained(self.model_id)
                self.model = self.model.to(device)
                if config.freeze_backbone:
                    for param in self.model.parameters():
                        param.requires_grad = False
                    self.model.eval()
                self.emulator_active = False
            except Exception as e:
                print(f"Warning: Failed to load real ProtT5 model. Falling back to emulator: {e}")
                self.emulator_active = True
        else:
            self.emulator_active = True

    def encode(self, sequence: str) -> Dict[str, torch.Tensor]:
        if self.emulator_active:
            token_embs, seq_emb = DeterministicProteinEmulator.generate(sequence, self.embedding_dim)
            return {
                "residue_embeddings": token_embs.to(self.device),
                "sequence_embedding": seq_emb.to(self.device),
                "is_emulator": torch.tensor(True, device=self.device)
            }
            
        # ProtT5 expects spaces between residues
        spaced_sequence = " ".join(list(sequence))
        # Replace non-standard amino acids
        spaced_sequence = re.sub(r"[UZOB]", "X", spaced_sequence)
        
        inputs = self.tokenizer(spaced_sequence, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
        hidden_states = outputs.hidden_states
        
        if self.config.layer_selection == "last4":
            selected_embs = torch.stack(hidden_states[-4:], dim=0).mean(dim=0)[0]
        else:
            selected_embs = hidden_states[-1][0]
            
        # Remove <eos> token at the end
        seq_len = len(sequence)
        residue_embs = selected_embs[:seq_len]
        seq_emb = residue_embs.mean(dim=0)
        
        return {
            "residue_embeddings": residue_embs,
            "sequence_embedding": seq_emb,
            "is_emulator": torch.tensor(False, device=self.device)
        }


# --- Cache Loader & Validator implementation ---

def get_sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()

def get_tensor_checksum(tensor: torch.Tensor) -> float:
    return float(torch.sum(torch.abs(tensor)).item())

def load_with_cache(pdb_id: str, sequence: str, config: ProteinFoundationConfig, encoder: FoundationModel) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """
    Loads pretrained protein embeddings from disk cache if valid, or extracts and caches them.
    
    Returns:
        residue_embs: Tensor of shape (L, D)
        seq_emb: Tensor of shape (D)
        is_emulator: Boolean indicating if fallback emulator was active
    """
    cache_dir = Path(config.cache_directory)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    pt_path = cache_dir / f"{pdb_id}_embedding.pt"
    json_path = cache_dir / f"{pdb_id}_metadata.json"
    
    seq_hash = get_sequence_hash(sequence)
    
    # 1. Try to load from cache
    if pt_path.exists() and json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                
            # Validate metadata fields
            valid = (
                meta.get("cache_schema_version") == config.cache_schema_version and
                meta.get("embedding_model") == config.protein_model and
                meta.get("projection_dim") == config.projection_dimension and
                meta.get("sequence_hash") == seq_hash
            )
            
            if valid:
                data = torch.load(pt_path, map_location="cpu")
                residue_embs = data["residue_embeddings"]
                seq_emb = data["sequence_embedding"]
                is_emulator = data.get("is_emulator", False)
                
                # Checksum validation
                checksum_ok = (abs(get_tensor_checksum(residue_embs) - meta.get("checksum", 0.0)) < 1e-3)
                
                if checksum_ok:
                    return residue_embs, seq_emb, is_emulator
        except Exception as e:
            print(f"Warning: Cache read failed for {pdb_id}: {e}. Regenerating...")
            
    # 2. Extract representation from scratch
    res_dict = encoder.encode(sequence)
    residue_embs = res_dict["residue_embeddings"].cpu()
    seq_emb = res_dict["sequence_embedding"].cpu()
    is_emulator = res_dict["is_emulator"].item()
    
    # 3. Write to cache
    try:
        torch.save({
            "residue_embeddings": residue_embs,
            "sequence_embedding": seq_emb,
            "is_emulator": is_emulator
        }, pt_path)
        
        meta = {
            "cache_schema_version": config.cache_schema_version,
            "embedding_model": config.protein_model,
            "embedding_revision": encoder.model_id if not is_emulator else "emulator",
            "projection_dim": config.projection_dimension,
            "sequence_hash": seq_hash,
            "created_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "checksum": get_tensor_checksum(residue_embs)
        }
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4)
    except Exception as e:
        print(f"Warning: Failed to write cache for {pdb_id}: {e}")
        
    return residue_embs, seq_emb, is_emulator

