from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class MCDropoutConfig:
    mc_samples: int = 30
    dropout_rate: Optional[float] = None  # None means inherit from base model regressor/attention

@dataclass
class EnsembleConfig:
    ensemble_size: int = 5
    checkpoint_paths: Optional[List[str]] = None

@dataclass
class EvidentialConfig:
    # Hyperparameters for Normal-Inverse-Gamma distribution
    # Prior strengths:
    lambda_val: float = 1.0  # prior virtual observations (mean confidence)
    alpha_val: float = 2.0   # prior shape parameter (variance confidence)
    beta_val: float = 1.0    # prior scale parameter (variance scaling)
    evidential_loss_weight: float = 0.1  # coefficient for scaling regularizer

@dataclass
class CalibrationConfig:
    calibration_method: str = "none"  # "none", "temperature_scaling", "variance_scaling"
    temperature: float = 1.0
    variance_multiplier: float = 1.0

@dataclass
class UncertaintyConfig:
    method: str = "mc_dropout"  # "mc_dropout", "ensemble", "evidential"
    mc_dropout: MCDropoutConfig = field(default_factory=MCDropoutConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    evidential: EvidentialConfig = field(default_factory=EvidentialConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
