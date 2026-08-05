from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class IntegratedGradientsConfig:
    steps: int = 50
    baseline_type: str = "zero"  # zero, random, average
    internal_batch_size: Optional[int] = None

@dataclass
class AttentionRolloutConfig:
    normalize: bool = True
    layer_reduction: str = "mean"  # mean, last, rollout

@dataclass
class OcclusionConfig:
    window_size: int = 1
    stride: int = 1
    fill_value: float = 0.0

@dataclass
class GraphAttributionConfig:
    sparsity_threshold: float = 0.1
    top_k: int = 10

@dataclass
class ExplainabilityConfig:
    method: str = "integrated_gradients"  # integrated_gradients, attention_rollout, occlusion, graph_attribution
    integrated_gradients: IntegratedGradientsConfig = field(default_factory=IntegratedGradientsConfig)
    attention_rollout: AttentionRolloutConfig = field(default_factory=AttentionRolloutConfig)
    occlusion: OcclusionConfig = field(default_factory=OcclusionConfig)
    graph_attribution: GraphAttributionConfig = field(default_factory=GraphAttributionConfig)
    agreement_metrics: List[str] = field(default_factory=lambda: ["spearman", "kendall", "jaccard", "top_k"])
    seed: int = 42
