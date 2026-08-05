from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class InteractionConfig:
    """Configuration class for the Interaction Representation Module (IRM)."""
    schema_version: str = "1.0"
    interaction_version: str = "1.0"
    feature_schema_version: str = "1.0"
    
    # Model dimensions
    interaction_dim: int = 32
    latent_dim: int = 32
    geometry_dim: int = 32
    chemistry_dim: int = 32
    representation_dim: int = 32
    
    # Hyperparameters
    num_rbf: int = 32
    cutoff_distance: float = 12.0
    learnable_scale: bool = True
    initial_alpha: float = 0.1
    normalize_bias: bool = True
    num_heads: int = 4
    dropout: float = 0.1
    activation: str = "relu"
    normalization: str = "layer"
    
    # Reserved backend option
    cache_backend: str = "none"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "interaction_version": self.interaction_version,
            "feature_schema_version": self.feature_schema_version,
            "interaction_dim": self.interaction_dim,
            "latent_dim": self.latent_dim,
            "geometry_dim": self.geometry_dim,
            "chemistry_dim": self.chemistry_dim,
            "representation_dim": self.representation_dim,
            "num_rbf": self.num_rbf,
            "cutoff_distance": self.cutoff_distance,
            "learnable_scale": self.learnable_scale,
            "initial_alpha": self.initial_alpha,
            "normalize_bias": self.normalize_bias,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "activation": self.activation,
            "normalization": self.normalization,
            "cache_backend": self.cache_backend,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InteractionConfig":
        if not d:
            return cls()
        return cls(
            schema_version=d.get("schema_version", "1.0"),
            interaction_version=d.get("interaction_version", "1.0"),
            feature_schema_version=d.get("feature_schema_version", "1.0"),
            interaction_dim=d.get("interaction_dim", 32),
            latent_dim=d.get("latent_dim", 32),
            geometry_dim=d.get("geometry_dim", 32),
            chemistry_dim=d.get("chemistry_dim", 32),
            representation_dim=d.get("representation_dim", 32),
            num_rbf=d.get("num_rbf", 32),
            cutoff_distance=d.get("cutoff_distance", 12.0),
            learnable_scale=d.get("learnable_scale", True),
            initial_alpha=d.get("initial_alpha", 0.1),
            normalize_bias=d.get("normalize_bias", True),
            num_heads=d.get("num_heads", 4),
            dropout=d.get("dropout", 0.1),
            activation=d.get("activation", "relu"),
            normalization=d.get("normalization", "layer"),
            cache_backend=d.get("cache_backend", "none"),
        )
