from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class PerturbationConfig:
    coordinate_noise_sigmas: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.25, 0.5, 1.0])
    feature_noise_fractions: List[float] = field(default_factory=lambda: [0.0, 0.05, 0.1, 0.2, 0.3])
    masking_fractions: List[float] = field(default_factory=lambda: [0.0, 0.05, 0.1, 0.2])
    edge_dropout_fractions: List[float] = field(default_factory=lambda: [0.0, 0.05, 0.1, 0.2])
    pocket_only: bool = True

@dataclass
class AdversarialConfig:
    epsilon: float = 0.05
    steps: int = 10
    pgd: bool = True

@dataclass
class OODConfig:
    split_strategies: List[str] = field(default_factory=lambda: ["scaffold", "protein_family"])

@dataclass
class RobustnessConfig:
    seed: int = 42
    perturbations: PerturbationConfig = field(default_factory=PerturbationConfig)
    adversarial: AdversarialConfig = field(default_factory=AdversarialConfig)
    ood: OODConfig = field(default_factory=OODConfig)
    weights: Dict[str, float] = field(default_factory=lambda: {
        "prediction_stability": 0.3,
        "calibration_stability": 0.2,
        "explanation_stability": 0.2,
        "representation_stability": 0.1,
        "ood_performance": 0.2
    })
