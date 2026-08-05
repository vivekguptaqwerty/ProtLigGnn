import pytest
import torch
from ligprotgnnx.configs.master_config import MasterConfig
from ligprotgnnx.inference.pipeline import UnifiedInferencePipeline
from tests.test_robustness import MockBaseModel, create_mock_batches

def test_pipeline_regression_identical():
    base_model = MockBaseModel()
    config = MasterConfig()
    pipeline = UnifiedInferencePipeline(base_model, config)
    
    l_batch, p_batch = create_mock_batches(batch_size=1)
    
    # 1. Pipeline output
    res = pipeline.predict(l_batch, p_batch)
    
    # 2. Independent run
    base_model.eval()
    with torch.no_grad():
        direct_pred = base_model(l_batch, p_batch)
        
    # Assert exact floating point equality
    assert torch.allclose(res.affinity, direct_pred, atol=1e-6)
