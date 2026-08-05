import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional

@dataclass
class ModelConfig:
    hidden_dim: int = 128
    dropout: float = 0.2
    pooling: str = "mean"

@dataclass
class UncertaintyConfig:
    method: str = "mc_dropout"  # mc_dropout, ensemble, evidential
    mc_samples: int = 30
    confidence_alpha: float = 0.05

@dataclass
class ExplainabilityConfig:
    method: str = "integrated_gradients"  # integrated_gradients, attention_rollout, occlusion, graph_attribution
    ig_steps: int = 50

@dataclass
class RobustnessConfig:
    coordinate_noise_sigma: float = 0.1
    feature_noise_fraction: float = 0.1

@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_file: str = "ligprotgnnx.log"

@dataclass
class MasterConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_json(self, filepath: str) -> None:
        with open(filepath, "w") as f:
            json.dump(asdict(self), f, indent=4)

    @classmethod
    def from_json(cls, filepath: str) -> "MasterConfig":
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(
            model=ModelConfig(**data.get("model", {})),
            uncertainty=UncertaintyConfig(**data.get("uncertainty", {})),
            explainability=ExplainabilityConfig(**data.get("explainability", {})),
            robustness=RobustnessConfig(**data.get("robustness", {})),
            logging=LoggingConfig(**data.get("logging", {}))
        )
