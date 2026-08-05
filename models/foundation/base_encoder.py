import torch
import torch.nn as nn
from typing import Dict, Any

class FoundationModel(nn.Module):
    """
    Abstract interface for all pretrained foundation encoders.
    All custom transformer adapters (ESM, ProtT5, etc.) must inherit from this class.
    """
    
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        super().__init__()
        self.model_name = model_name
        self.device = device
        
    def encode(self, sequence: str) -> Dict[str, torch.Tensor]:
        """
        Extract token-level and sequence-level embeddings from the raw sequence.
        
        Args:
            sequence: Raw amino acid sequence string.
            
        Returns:
            Dict containing:
                - 'residue_embeddings': torch.Tensor of shape (seq_len, embedding_dim)
                - 'sequence_embedding': torch.Tensor of shape (embedding_dim)
        """
        raise NotImplementedError("Each subclass of FoundationModel must implement the encode() method.")
