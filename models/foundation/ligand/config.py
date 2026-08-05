from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class LigandFoundationConfig:
    ligand_model: str = "chemberta"         # "chemberta", "molformer", "none"
    latent_dimension: int = 256             # Dimension of baseline representations
    projection_dimension: int = 256        # Output dimension of the projection head
    freeze_backbone: bool = True
    layer_selection: str = "last"
    pooling_strategy: str = "graph_node_mapping"  # "graph_node_mapping", "mean", "cls"
    cache_schema_version: str = "1.0"
    checkpoint_version: str = "1.0"
    schema_version: str = "1.0"
    cache_directory: str = "embedding_cache/ligand"

    def __post_init__(self) -> None:
        if self.ligand_model not in ["chemberta", "molformer", "none"]:
            raise ValueError(f"Invalid ligand_model: {self.ligand_model}")
        if self.pooling_strategy not in ["graph_node_mapping", "mean", "cls"]:
            raise ValueError(f"Invalid pooling_strategy: {self.pooling_strategy}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ligand_model": self.ligand_model,
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
    def from_dict(cls, d: Dict[str, Any]) -> "LigandFoundationConfig":
        return cls(
            ligand_model=d.get("ligand_model", "chemberta"),
            latent_dimension=d.get("latent_dimension", 256),
            projection_dimension=d.get("projection_dimension", 256),
            freeze_backbone=d.get("freeze_backbone", True),
            layer_selection=d.get("layer_selection", "last"),
            pooling_strategy=d.get("pooling_strategy", "graph_node_mapping"),
            cache_schema_version=d.get("cache_schema_version", "1.0"),
            checkpoint_version=d.get("checkpoint_version", "1.0"),
            schema_version=d.get("schema_version", "1.0"),
            cache_directory=d.get("cache_directory", "embedding_cache/ligand")
        )
