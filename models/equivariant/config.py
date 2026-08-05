from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class EquivariantConfig:
    equivariant_model: str = "egnn"
    num_layers: int = 4
    hidden_dim: int = 256
    edge_dim: int = 32
    dropout: float = 0.1
    coordinate_updates: bool = True
    residual: bool = True
    layer_norm: bool = True
    schema_version: str = "1.0"
    checkpoint_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.equivariant_model not in ["egnn"]:
            raise ValueError(f"Invalid equivariant_model: {self.equivariant_model}")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "equivariant_model": self.equivariant_model,
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "edge_dim": self.edge_dim,
            "dropout": self.dropout,
            "coordinate_updates": self.coordinate_updates,
            "residual": self.residual,
            "layer_norm": self.layer_norm,
            "schema_version": self.schema_version,
            "checkpoint_version": self.checkpoint_version
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EquivariantConfig":
        return cls(
            equivariant_model=d.get("equivariant_model", "egnn"),
            num_layers=d.get("num_layers", 4),
            hidden_dim=d.get("hidden_dim", 256),
            edge_dim=d.get("edge_dim", 32),
            dropout=d.get("dropout", 0.1),
            coordinate_updates=d.get("coordinate_updates", True),
            residual=d.get("residual", True),
            layer_norm=d.get("layer_norm", True),
            schema_version=d.get("schema_version", "1.0"),
            checkpoint_version=d.get("checkpoint_version", "1.0")
        )
