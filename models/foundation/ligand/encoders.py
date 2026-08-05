import os
import hashlib
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple

from models.foundation.base_encoder import FoundationModel
from models.foundation.ligand.config import LigandFoundationConfig

try:
    import transformers
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class DeterministicLigandEmulator:
    """
    Deterministic molecular embedding emulator for testing.
    Generates deterministic features based on SMILES string character hashing.
    """
    
    @staticmethod
    def generate(smiles: str, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate stable seed from SMILES
        hasher = hashlib.sha256(smiles.encode("utf-8"))
        seed = int(hasher.hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        
        # Tokenize by characters as a simple fallback sequence length
        seq_len = len(smiles)
        token_embs = rng.normal(0.0, 1.0, size=(seq_len, dim)).astype(np.float32)
        mol_emb = np.mean(token_embs, axis=0)
        
        return torch.from_numpy(token_embs), torch.from_numpy(mol_emb)


class ChemBERTaEncoder(FoundationModel):
    """
    Adapter for the ChemBERTa molecular encoder (seyonec/ChemBERTa_zinc250k_v2_pruned).
    """
    
    def __init__(self, config: LigandFoundationConfig, device: str = "cpu") -> None:
        super().__init__(model_name="chemberta", device=device)
        self.config = config
        self.model_id = "seyonec/ChemBERTa_zinc250k_v2_pruned"
        self.embedding_dim = 384
        self.emulator_active = True
        
        use_real = (os.environ.get("LIGPROTGNN_PRODUCTION", "0") == "1")
        
        if TRANSFORMERS_AVAILABLE and use_real:
            try:
                print(f"Loading ChemBERTa transformer ({self.model_id})...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self.model = AutoModel.from_pretrained(self.model_id)
                self.model = self.model.to(device)
                if config.freeze_backbone:
                    for param in self.model.parameters():
                        param.requires_grad = False
                    self.model.eval()
                self.emulator_active = False
            except Exception as e:
                print(f"Warning: Failed to load ChemBERTa. Falling back to emulator: {e}")
                self.emulator_active = True
        else:
            self.emulator_active = True

    def encode(self, smiles: str) -> Dict[str, torch.Tensor]:
        if self.emulator_active:
            token_embs, mol_emb = DeterministicLigandEmulator.generate(smiles, self.embedding_dim)
            return {
                "atom_embeddings": token_embs.to(self.device),
                "molecule_embedding": mol_emb.to(self.device),
                "is_emulator": torch.tensor(True, device=self.device)
            }
            
        inputs = self.tokenizer(smiles, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
        hidden_states = outputs.hidden_states
        selected_embs = hidden_states[-1][0] # (seq_len_with_special_tokens, dim)
        
        # Strip special tokens (CLS and SEP)
        # ChemBERTa uses standard BERT format: [CLS] tokens [SEP]
        seq_len = len(inputs["input_ids"][0])
        atom_embs = selected_embs[1:seq_len - 1]
        
        mol_emb = atom_embs.mean(dim=0)
        
        return {
            "atom_embeddings": atom_embs,
            "molecule_embedding": mol_emb,
            "is_emulator": torch.tensor(False, device=self.device)
        }


class MolFormerEncoder(FoundationModel):
    """
    Adapter for the MolFormer molecular encoder (ibm/MoLFormer-XL-ClowM50_10pct).
    """
    
    def __init__(self, config: LigandFoundationConfig, device: str = "cpu") -> None:
        super().__init__(model_name="molformer", device=device)
        self.config = config
        self.model_id = "ibm/MoLFormer-XL-ClowM50_10pct"
        self.embedding_dim = 768
        self.emulator_active = True
        
        use_real = (os.environ.get("LIGPROTGNN_PRODUCTION", "0") == "1")
        
        if TRANSFORMERS_AVAILABLE and use_real:
            try:
                print(f"Loading MolFormer transformer ({self.model_id})...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
                self.model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True)
                self.model = self.model.to(device)
                if config.freeze_backbone:
                    for param in self.model.parameters():
                        param.requires_grad = False
                    self.model.eval()
                self.emulator_active = False
            except Exception as e:
                print(f"Warning: Failed to load MolFormer. Falling back to emulator: {e}")
                self.emulator_active = True
        else:
            self.emulator_active = True

    def encode(self, smiles: str) -> Dict[str, torch.Tensor]:
        if self.emulator_active:
            token_embs, mol_emb = DeterministicLigandEmulator.generate(smiles, self.embedding_dim)
            return {
                "atom_embeddings": token_embs.to(self.device),
                "molecule_embedding": mol_emb.to(self.device),
                "is_emulator": torch.tensor(True, device=self.device)
            }
            
        inputs = self.tokenizer(smiles, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
        hidden_states = outputs.hidden_states
        selected_embs = hidden_states[-1][0]
        
        seq_len = len(inputs["input_ids"][0])
        # Strip CLS and SEP tokens
        atom_embs = selected_embs[1:seq_len - 1]
        
        mol_emb = atom_embs.mean(dim=0)
        
        return {
            "atom_embeddings": atom_embs,
            "molecule_embedding": mol_emb,
            "is_emulator": torch.tensor(False, device=self.device)
        }
