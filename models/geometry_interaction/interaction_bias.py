import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict
from models.geometry_interaction.interfaces import InteractionFusion, InteractionContext
from models.geometry_interaction.interaction_encoder import (
    GeometryFeatureEncoder, ChemistryFeatureEncoder, RepresentationFeatureEncoder
)
from models.geometry_interaction.config import InteractionConfig

class ConcatFusion(InteractionFusion):
    """Concatenation fusion combining geometric, chemical, and latent features."""
    def __init__(self, geo_dim: int, chem_dim: int, rep_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.projector = nn.Linear(geo_dim + chem_dim + rep_dim, latent_dim)

    def forward(self, features: Dict[str, Tensor]) -> Tensor:
        geo = features["geometry"]
        chem = features["chemistry"]
        rep = features["representation"]
        
        fused = torch.cat([geo, chem, rep], dim=-1)  # [L, P, geo_dim + chem_dim + rep_dim]
        return self.projector(fused)  # [L, P, latent_dim]


class InteractionBias(nn.Module):
    """Main interaction representation module coordinating encoders, fusion, and projection."""
    def __init__(self, config: InteractionConfig, hidden_dim: int = 128) -> None:
        super().__init__()
        self.config = config
        
        # Encoders
        self.geometry_encoder = GeometryFeatureEncoder(
            num_basis=config.num_rbf,
            start=0.0,
            stop=config.cutoff_distance,
            geometry_dim=config.geometry_dim
        )
        self.chemistry_encoder = ChemistryFeatureEncoder(
            chemistry_dim=config.chemistry_dim
        )
        self.representation_encoder = RepresentationFeatureEncoder(
            hidden_dim=hidden_dim,
            representation_dim=config.representation_dim
        )
        
        # Fusion
        self.fusion = ConcatFusion(
            geo_dim=config.geometry_dim,
            chem_dim=config.chemistry_dim,
            rep_dim=config.representation_dim,
            latent_dim=config.latent_dim
        )
        
        # Interaction Latent Space processing sequence
        self.latent_proj = nn.Linear(config.latent_dim, config.interaction_dim)
        self.latent_norm = nn.LayerNorm(config.interaction_dim)
        self.latent_act = nn.ReLU()
        self.latent_drop = nn.Dropout(config.dropout)
        
        # Bias projection
        self.bias_projector = nn.Linear(config.interaction_dim, config.num_heads)
        
        # Cache slot for hookless access
        self.latest_interaction_emb = None

    def forward(self, context: InteractionContext) -> Tensor:
        # 1. Forward pass on branches
        features = {
            "geometry": self.geometry_encoder(context),
            "chemistry": self.chemistry_encoder(context),
            "representation": self.representation_encoder(context)
        }
        
        # 2. Fuse branch embeddings
        fused = self.fusion(features)  # [L, P, latent_dim]
        
        # 3. Latent space processing: Linear -> LayerNorm -> ReLU -> Dropout
        latent = self.latent_proj(fused)
        latent = self.latent_norm(latent)
        latent = self.latent_act(latent)
        interaction_emb = self.latent_drop(latent)  # [L, P, interaction_dim]
        
        # Cache for diagnostics and explainability
        self.latest_interaction_emb = interaction_emb.detach()
        
        # 4. Project to attention bias logits
        bias = self.bias_projector(interaction_emb)  # [L, P, num_heads]
        
        # Permute to [num_heads, L, P]
        return bias.permute(2, 0, 1)
