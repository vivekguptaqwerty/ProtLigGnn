import os
import torch
import torch.nn as nn
import hashlib
from typing import List, Dict, Optional

class ESM2FeatureExtractor(nn.Module):
    """
    ESM-2 Protein Language Model (PLM) Feature Extractor for LigProtGNN-X v5.0.
    Maps 20 amino acid residue sequences or sequence IDs to 1280-dimensional PLM embeddings.
    Features automatic disk caching in `embedding_cache/esm2/` for fast training and inference.
    """
    def __init__(self, cache_dir: str = "embedding_cache/esm2", embed_dim: int = 1280, device: str = "cpu"):
        super(ESM2FeatureExtractor, self).__init__()
        self.cache_dir = cache_dir
        self.embed_dim = embed_dim
        self.device = device
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Fallback learned embedding projection (20 AA -> 1280 PLM dimension)
        self.aa_projection = nn.Sequential(
            nn.Linear(20, 256),
            nn.SiLU(),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        self.esm_model = None
        self.alphabet = None

    def _hash_sequence(self, sequence: str) -> str:
        return hashlib.sha256(sequence.encode('utf-8')).hexdigest()[:16]

    def load_esm_model(self, model_name: str = "esm2_t33_650M_UR50D"):
        """Attempts loading full ESM-2 PyTorch model from torch.hub if available."""
        try:
            import esm
            self.esm_model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.esm_model = self.esm_model.to(self.device).eval()
            print(f"Successfully initialized ESM-2 PLM model ({model_name}).")
        except Exception as e:
            print(f"ESM-2 model loading deferred (using neural projection fallback): {e}")

    def extract_sequence_embedding(self, sequence: str) -> torch.Tensor:
        """
        Extracts or loads cached 1280-dim residue embeddings for a protein sequence.
        Returns tensor of shape [L, 1280].
        """
        seq_hash = self._hash_sequence(sequence)
        cache_path = os.path.join(self.cache_dir, f"{seq_hash}.pt")
        
        if os.path.exists(cache_path):
            try:
                return torch.load(cache_path, map_location='cpu')
            except Exception:
                pass
                
        if self.esm_model is not None and self.alphabet is not None:
            batch_converter = self.alphabet.get_batch_converter()
            data = [("protein", sequence)]
            _, _, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(self.device)
            
            with torch.no_grad():
                results = self.esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
                token_representations = results["representations"][33][0, 1:-1].cpu() # Exclude BOS/EOS
                
            torch.save(token_representations, cache_path)
            return token_representations

        # Fallback projection from 20-dim one-hot vectors
        aa_map = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        one_hots = torch.zeros(len(sequence), 20)
        for idx, char in enumerate(sequence.upper()):
            if char in aa_map:
                one_hots[idx, aa_map[char]] = 1.0
            else:
                one_hots[idx] = 0.05
                
        with torch.no_grad():
            embeddings = self.aa_projection(one_hots)
            
        torch.save(embeddings, cache_path)
        return embeddings

    def forward(self, residue_one_hots: torch.Tensor) -> torch.Tensor:
        """
        Differentiable forward projection for node residue tensors [N, 20] -> [N, 1280].
        """
        return self.aa_projection(residue_one_hots)
