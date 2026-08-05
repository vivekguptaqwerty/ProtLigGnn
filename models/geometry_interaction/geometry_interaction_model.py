import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool, global_max_pool

# Import certified baseline components
from protliggnn_train import LigandEncoder, ProteinEncoder
from models.geometry_interaction.config import InteractionConfig
from models.geometry_interaction.interfaces import InteractionContext
from models.geometry_interaction.interaction_bias import InteractionBias

class InteractionBiasedCrossAttention(nn.Module):
    """Bidirectional Cross-Attention layer incorporating pluggable Interaction Representations."""
    def __init__(self, hidden_dim: int, config: InteractionConfig, dropout: float = 0.1,
                 use_residual: bool = True, use_layernorm: bool = True) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.config = config
        
        self.ligand_to_protein_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=config.num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.protein_to_ligand_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=config.num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        self.ligand_norm = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
        self.protein_norm = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
        
        # Pluggable IRM interaction bias module
        self.attention_bias = InteractionBias(config, hidden_dim=hidden_dim)
        
        # Learnable scale parameter in log-space (exp(log_alpha) > 0)
        self.learnable_scale = config.learnable_scale
        if self.learnable_scale:
            self.log_alpha = nn.Parameter(torch.tensor(math.log(config.initial_alpha)))
        else:
            self.register_buffer("log_alpha", torch.tensor(math.log(config.initial_alpha)))

    def forward(self, ligand_x: torch.Tensor, protein_x: torch.Tensor,
                ligand_pos: torch.Tensor, protein_pos: torch.Tensor,
                ligand_x_raw: torch.Tensor, protein_x_raw: torch.Tensor,
                ligand_batch: torch.Tensor, protein_batch: torch.Tensor,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Create InteractionContext
        context = InteractionContext(
            version="1.0",
            ligand_pos=ligand_pos,
            protein_pos=protein_pos,
            ligand_x_raw=ligand_x_raw,
            protein_x_raw=protein_x_raw,
            ligand_x_enc=ligand_x,
            protein_x_enc=protein_x,
            ligand_batch=ligand_batch,
            protein_batch=protein_batch
        )

        # 2. Get attention bias from IRM
        raw_bias = self.attention_bias(context)  # [num_heads, L, P]
        
        # 3. Apply normalization
        if self.config.normalize_bias:
            raw_bias = F.layer_norm(raw_bias, raw_bias.shape[1:])
            
        # 4. Apply scale parameter
        alpha = torch.exp(self.log_alpha)
        bias_l2p = alpha * raw_bias
        bias_p2l = bias_l2p.permute(0, 2, 1)  # [num_heads, P, L]
        
        ligand_seq = ligand_x.unsqueeze(0)
        protein_seq = protein_x.unsqueeze(0)
        
        # 5. Cross-attention forward
        ligand_context, _ = self.ligand_to_protein_attn(
            query=ligand_seq,
            key=protein_seq,
            value=protein_seq,
            need_weights=False,
            attn_mask=bias_l2p,
            **kwargs
        )
        protein_context, _ = self.protein_to_ligand_attn(
            query=protein_seq,
            key=ligand_seq,
            value=ligand_seq,
            need_weights=False,
            attn_mask=bias_p2l,
            **kwargs
        )
        
        # 6. Residual connections and norm
        l_out = ligand_context + ligand_seq if self.use_residual else ligand_context
        p_out = protein_context + protein_seq if self.use_residual else protein_context
        
        ligand_context = self.ligand_norm(l_out.squeeze(0))
        protein_context = self.protein_norm(p_out.squeeze(0))
        return ligand_context, protein_context


class ProtLigGNNGeometryInteraction(nn.Module):
    """ProtLigGNN model upgraded with the Interaction Representation Module (IRM)."""
    def __init__(self, ligand_dim: int, protein_dim: int, hidden_dim: int = 128,
                 no_crossgraph: bool = False, dropout: float = 0.2,
                 use_residual: bool = True, use_layernorm: bool = True,
                 use_attention: bool = True, pooling: str = "mean",
                 config: InteractionConfig = None) -> None:
        super().__init__()
        self.no_crossgraph = no_crossgraph
        self.use_attention = use_attention
        self.hidden_dim = hidden_dim
        self.pooling = pooling
        self.config = config if config is not None else InteractionConfig()
        
        # Coordinate-free encoders (Baseline v1.1 compliant)
        self.ligand_encoder = LigandEncoder(
            ligand_dim, hidden_dim, dropout=dropout * 0.5, use_layernorm=use_layernorm
        )
        self.protein_encoder = ProteinEncoder(
            protein_dim, hidden_dim, use_layernorm=use_layernorm
        )
        
        # Pluggable Interaction-Aware Cross-Attention
        self.cross_attention = InteractionBiasedCrossAttention(
            hidden_dim=hidden_dim,
            config=self.config,
            dropout=dropout * 0.5,
            use_residual=use_residual,
            use_layernorm=use_layernorm
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Regressor head (Baseline v1.1 compliant)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )

    def cross_graph_interaction(self, ligand_x: torch.Tensor, protein_x: torch.Tensor,
                               ligand_pos: torch.Tensor, protein_pos: torch.Tensor,
                               ligand_x_raw: torch.Tensor, protein_x_raw: torch.Tensor,
                               ligand_batch: torch.Tensor, protein_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.no_crossgraph:
            zero_l = torch.zeros_like(ligand_x)
            zero_p = torch.zeros_like(protein_x)
            return torch.cat([ligand_x, zero_l], dim=-1), torch.cat([protein_x, zero_p], dim=-1)

        if not self.use_attention:
            return torch.cat([ligand_x, ligand_x], dim=-1), torch.cat([protein_x, protein_x], dim=-1)

        ligand_context, protein_context = self.cross_attention(
            ligand_x=ligand_x,
            protein_x=protein_x,
            ligand_pos=ligand_pos,
            protein_pos=protein_pos,
            ligand_x_raw=ligand_x_raw,
            protein_x_raw=protein_x_raw,
            ligand_batch=ligand_batch,
            protein_batch=protein_batch
        )
        return (
            torch.cat([ligand_x, ligand_context], dim=-1),
            torch.cat([protein_x, protein_context], dim=-1),
        )

    def forward(self, ligand_batch: Batch, protein_batch: Batch) -> torch.Tensor:
        # 1. Encode GNN representations
        ligand_x = self.ligand_encoder(ligand_batch.x, ligand_batch.edge_index)
        protein_x = self.protein_encoder(protein_batch.x, protein_batch.edge_index)
        
        # 2. Dynamic batch looping
        ligand_chunks = []
        protein_chunks = []
        batch_size = ligand_batch.num_graphs
        for idx in range(batch_size):
            ligand_mask = ligand_batch.batch == idx
            protein_mask = protein_batch.batch == idx
            
            ligand_chunk, protein_chunk = self.cross_graph_interaction(
                ligand_x=ligand_x[ligand_mask],
                protein_x=protein_x[protein_mask],
                ligand_pos=ligand_batch.pos[ligand_mask],
                protein_pos=protein_batch.pos[protein_mask],
                ligand_x_raw=ligand_batch.x[ligand_mask],
                protein_x_raw=protein_batch.x[protein_mask],
                ligand_batch=ligand_batch.batch[ligand_mask],
                protein_batch=protein_batch.batch[protein_mask]
            )
            ligand_chunks.append(ligand_chunk)
            protein_chunks.append(protein_chunk)
            
        ligand_joint = torch.cat(ligand_chunks, dim=0)
        protein_joint = torch.cat(protein_chunks, dim=0)
        
        # 3. Readout and pooling
        if self.pooling == "max":
            ligand_pool = global_max_pool(ligand_joint, ligand_batch.batch)
            protein_pool = global_max_pool(protein_joint, protein_batch.batch)
        else:
            ligand_pool = global_mean_pool(ligand_joint, ligand_batch.batch)
            protein_pool = global_mean_pool(protein_joint, protein_batch.batch)
            
        joint = self.dropout(torch.cat([ligand_pool, protein_pool], dim=-1))
        return self.regressor(joint).squeeze(-1)
