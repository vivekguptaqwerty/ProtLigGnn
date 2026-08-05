from dataclasses import dataclass, asdict
from typing import Literal, Dict, Any

@dataclass
class AttentionBiasConfig:
    """Configuration class for the pluggable attention bias subsystem."""
    bias_type: Literal["rbf", "learned"] = "rbf"
    num_rbf: int = 32
    cutoff_distance: float = 12.0
    learnable_scale: bool = True
    initial_alpha: float = 0.1
    normalize_bias: bool = True
    num_heads: int = 4

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the configuration to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AttentionBiasConfig":
        """Instantiates the configuration from a dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
