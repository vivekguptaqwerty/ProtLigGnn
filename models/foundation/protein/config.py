import json
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class ProteinFoundationConfig:
    protein_model: str = "esm2"            # "esm2", "prot_t5", "none"
    latent_dimension: int = 256            # Dimension of baseline representations
    projection_dimension: int = 256       # Output dimension of the projection head
    freeze_backbone: bool = True
    layer_selection: str = "last"          # "last", "last4"
    pooling_strategy: str = "pocket_only"  # "pocket_only", "mean", "cls"
    cache_schema_version: str = "1.0"
    checkpoint_version: str = "1.0"
    schema_version: str = "1.0"
    cache_directory: str = "embedding_cache/protein"

    def __post_init__(self) -> None:
        if self.protein_model not in ["esm2", "prot_t5", "none"]:
            raise ValueError(f"Invalid protein_model: {self.protein_model}")
        if self.layer_selection not in ["last", "last4"]:
            raise ValueError(f"Invalid layer_selection: {self.layer_selection}")
        if self.pooling_strategy not in ["pocket_only", "mean", "cls"]:
            raise ValueError(f"Invalid pooling_strategy: {self.pooling_strategy}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protein_model": self.protein_model,
            "latent_dimension": self.latent_dimension,
            "projection_dimension": self.projection_dimension,
            "freeze_backbone": self.freeze_backbone,
            "layer_selection": self.layer_selection,
            "pooling_strategy": self.pooling_strategy,
            "cache_schema_version": self.cache_schema_version,
            "checkpoint_version": self.checkpoint_version,
            "schema_version": self.schema_version,
            "cache_directory": self.cache_directory
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProteinFoundationConfig":
        return cls(
            protein_model=d.get("protein_model", "esm2"),
            latent_dimension=d.get("latent_dimension", 256),
            projection_dimension=d.get("projection_dimension", 256),
            freeze_backbone=d.get("freeze_backbone", True),
            layer_selection=d.get("layer_selection", "last"),
            pooling_strategy=d.get("pooling_strategy", "pocket_only"),
            cache_schema_version=d.get("cache_schema_version", "1.0"),
            checkpoint_version=d.get("checkpoint_version", "1.0"),
            schema_version=d.get("schema_version", "1.0"),
            cache_directory=d.get("cache_directory", "embedding_cache/protein")
        )
