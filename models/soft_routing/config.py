import json
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class RoutingConfig:
    router_type: str = "gated"  # "linear", "residual", "gated"
    routing_level: str = "embedding"  # "embedding", "attention_head"
    latent_dimension: int = 256
    routing_dimension: int = 256
    num_layers: int = 2
    activation: str = "relu"  # "relu", "gelu"
    normalization: str = "layer"  # "layer", "none"
    dropout: float = 0.1
    gate_regularization: float = 0.0
    entropy_regularization: float = 0.01
    sparsity_regularization: float = 0.0
    diversity_regularization: float = 0.0
    learnable_temperature: bool = True
    router_capacity: str = "base"  # "tiny", "small", "base"
    checkpoint_version: str = "1.0"
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.router_type not in ["linear", "residual", "gated"]:
            raise ValueError(f"Invalid router_type: {self.router_type}")
        if self.routing_level not in ["embedding", "attention_head"]:
            raise ValueError(f"Invalid routing_level: {self.routing_level}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "router_type": self.router_type,
            "routing_level": self.routing_level,
            "latent_dimension": self.latent_dimension,
            "routing_dimension": self.routing_dimension,
            "num_layers": self.num_layers,
            "activation": self.activation,
            "normalization": self.normalization,
            "dropout": self.dropout,
            "gate_regularization": self.gate_regularization,
            "entropy_regularization": self.entropy_regularization,
            "sparsity_regularization": self.sparsity_regularization,
            "diversity_regularization": self.diversity_regularization,
            "learnable_temperature": self.learnable_temperature,
            "router_capacity": self.router_capacity,
            "checkpoint_version": self.checkpoint_version,
            "schema_version": self.schema_version
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RoutingConfig":
        return cls(
            router_type=d.get("router_type", "gated"),
            routing_level=d.get("routing_level", "embedding"),
            latent_dimension=d.get("latent_dimension", 256),
            routing_dimension=d.get("routing_dimension", 256),
            num_layers=d.get("num_layers", 2),
            activation=d.get("activation", "relu"),
            normalization=d.get("normalization", "layer"),
            dropout=d.get("dropout", 0.1),
            gate_regularization=d.get("gate_regularization", 0.0),
            entropy_regularization=d.get("entropy_regularization", 0.01),
            sparsity_regularization=d.get("sparsity_regularization", 0.0),
            diversity_regularization=d.get("diversity_regularization", 0.0),
            learnable_temperature=d.get("learnable_temperature", True),
            router_capacity=d.get("router_capacity", "base"),
            checkpoint_version=d.get("checkpoint_version", "1.0"),
            schema_version=d.get("schema_version", "1.0")
        )
