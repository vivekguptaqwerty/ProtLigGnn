import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class BidirectionalCrossAttention(nn.Module):
    def __init__(self, dim: int = 256, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Protein query, key, value projections
        self.q_proj_p = nn.Linear(dim, dim)
        self.k_proj_p = nn.Linear(dim, dim)
        self.v_proj_p = nn.Linear(dim, dim)
        
        # Ligand query, key, value projections
        self.q_proj_l = nn.Linear(dim, dim)
        self.k_proj_l = nn.Linear(dim, dim)
        self.v_proj_l = nn.Linear(dim, dim)
        
        # Out projections
        self.out_p = nn.Linear(dim, dim)
        self.out_l = nn.Linear(dim, dim)
        
        # Layer normalizations
        self.norm_p = nn.LayerNorm(dim)
        self.norm_l = nn.LayerNorm(dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Cache for attention visualization
        self.last_attn_p2l = None
        self.last_attn_l2p = None

    def forward(
        self, 
        protein_rep: torch.Tensor, 
        ligand_rep: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bidirectional Cross-Attention.
        
        Args:
            protein_rep: Tensor of shape (P, D)
            ligand_rep: Tensor of shape (L, D)
            
        Returns:
            protein_context: Shape (P, D)
            ligand_context: Shape (L, D)
        """
        P = protein_rep.size(0)
        L = ligand_rep.size(0)
        
        # Add batch dimension to conform to standard attention shapes (1, Seq, Dim)
        p_seq = protein_rep.unsqueeze(0)  # (1, P, D)
        l_seq = ligand_rep.unsqueeze(0)  # (1, L, D)
        
        # --- 1. Protein Attends to Ligand (P2L) ---
        q_p = self.q_proj_p(p_seq).view(1, P, self.num_heads, self.head_dim).transpose(1, 2)  # (1, H, P, d)
        k_l = self.k_proj_l(l_seq).view(1, L, self.num_heads, self.head_dim).transpose(1, 2)  # (1, H, L, d)
        v_l = self.v_proj_l(l_seq).view(1, L, self.num_heads, self.head_dim).transpose(1, 2)  # (1, H, L, d)
        
        scores_p2l = torch.matmul(q_p, k_l.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (1, H, P, L)
        attn_p2l = F.softmax(scores_p2l, dim=-1)
        self.last_attn_p2l = attn_p2l.detach().cpu()
        
        context_p = torch.matmul(self.dropout(attn_p2l), v_l)  # (1, H, P, d)
        context_p = context_p.transpose(1, 2).contiguous().view(1, P, self.dim)  # (1, P, D)
        context_p = self.norm_p(protein_rep + self.out_p(context_p.squeeze(0)))  # Residual + Norm
        
        # --- 2. Ligand Attends to Protein (L2P) ---
        q_l = self.q_proj_l(l_seq).view(1, L, self.num_heads, self.head_dim).transpose(1, 2)  # (1, H, L, d)
        k_p = self.k_proj_p(p_seq).view(1, P, self.num_heads, self.head_dim).transpose(1, 2)  # (1, H, P, d)
        v_p = self.v_proj_p(p_seq).view(1, P, self.num_heads, self.head_dim).transpose(1, 2)  # (1, H, P, d)
        
        scores_l2p = torch.matmul(q_l, k_p.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (1, H, L, P)
        attn_l2p = F.softmax(scores_l2p, dim=-1)
        self.last_attn_l2p = attn_l2p.detach().cpu()
        
        context_l = torch.matmul(self.dropout(attn_l2p), v_p)  # (1, H, L, d)
        context_l = context_l.transpose(1, 2).contiguous().view(1, L, self.dim)  # (1, L, D)
        context_l = self.norm_l(ligand_rep + self.out_l(context_l.squeeze(0)))  # Residual + Norm
        
        return context_p, context_l
