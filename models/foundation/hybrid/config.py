from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class HybridFoundationConfig:
    protein_model: str = "esm2"             # "esm2", "prot_t5", "none"
    ligand_model: str = "chemberta"         # "chemberta", "molformer", "none"
    fusion_strategy: str = "cross_attention"  # "cross_attention", "concatenation", "weighted_sum", "gated", "residual", "no_fusion"
    projection_dimension: int = 256        # Projected representation dimension
    cache_schema_version: str = "1.0"
    checkpoint_version: str = "1.0"
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.protein_model not in ["esm2", "prot_t5", "none"]:
            raise ValueError(f"Invalid protein_model: {self.protein_model}")
        if self.ligand_model not in ["chemberta", "molformer", "none"]:
            raise ValueError(f"Invalid ligand_model: {self.ligand_model}")
        if self.fusion_strategy not in ["cross_attention", "concatenation", "weighted_sum", "gated", "residual", "no_fusion"]:
            raise ValueError(f"Invalid fusion_strategy: {self.fusion_strategy}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protein_model": self.protein_model,
            "ligand_model": self.ligand_model,
            "fusion_strategy": self.fusion_strategy,
            "projection_dimension": self.projection_dimension,
            "cache_schema_version": self.cache_schema_version,
            "checkpoint_version": self.checkpoint_version,
            "schema_version": self.schema_version
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HybridFoundationConfig":
        return cls(
            protein_model=d.get("protein_model", "esm2"),
            ligand_model=d.get("ligand_model", "chemberta"),
            fusion_strategy=d.get("fusion_strategy", "cross_attention"),
            projection_dimension=d.get("projection_dimension", 256),
            cache_schema_version=d.get("cache_schema_version", "1.0"),
            checkpoint_version=d.get("checkpoint_version", "1.0"),
            schema_version=d.get("schema_version", "1.0")
        )
