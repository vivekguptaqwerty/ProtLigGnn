from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class PhysicsGuidedConfig:
    """Configuration schema for Physics-Guided Multitask Learning."""
    schema_version: str = "1.0"
    checkpoint_version: str = "1.0"
    
    # Model parameters
    contact_threshold: float = 4.0
    loss_weight: float = 0.1
    loss_weight_strategy: str = "fixed"  # fixed, learnable, cosine, linear
    contact_hidden_dim: int = 128
    dropout: float = 0.1
    activation: str = "relu"
    normalization: str = "layer"
    learnable_scale: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "checkpoint_version": self.checkpoint_version,
            "contact_threshold": self.contact_threshold,
            "loss_weight": self.loss_weight,
            "loss_weight_strategy": self.loss_weight_strategy,
            "contact_hidden_dim": self.contact_hidden_dim,
            "dropout": self.dropout,
            "activation": self.activation,
            "normalization": self.normalization,
            "learnable_scale": self.learnable_scale,
        }
        
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PhysicsGuidedConfig":
        if not d:
            return cls()
        return cls(
            schema_version=d.get("schema_version", "1.0"),
            checkpoint_version=d.get("checkpoint_version", "1.0"),
            contact_threshold=float(d.get("contact_threshold", 4.0)),
            loss_weight=float(d.get("loss_weight", 0.1)),
            loss_weight_strategy=d.get("loss_weight_strategy", "fixed"),
            contact_hidden_dim=int(d.get("contact_hidden_dim", 128)),
            dropout=float(d.get("dropout", 0.1)),
            activation=d.get("activation", "relu"),
            normalization=d.get("normalization", "layer"),
            learnable_scale=bool(d.get("learnable_scale", True)),
        )
