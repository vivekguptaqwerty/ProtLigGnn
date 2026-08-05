import pytest
import torch
from ligprotgnnx.configs.master_config import MasterConfig
from ligprotgnnx.inference.pipeline import UnifiedInferencePipeline
from tests.test_robustness import MockBaseModel, create_mock_batches

def test_prediction_numerical_stability():
    base_model = MockBaseModel()
    config = MasterConfig()
    pipeline = UnifiedInferencePipeline(base_model, config)
    
    l_batch, p_batch = create_mock_batches(batch_size=1)
    
    # Verify deterministic behavior across multiple inferences
    res1 = pipeline.predict(l_batch, p_batch)
    res2 = pipeline.predict(l_batch, p_batch)
    
    assert torch.equal(res1.affinity, res2.affinity)
    assert not torch.isnan(res1.affinity).any()
