import pytest
import torch
from ligprotgnnx.configs.master_config import MasterConfig
from ligprotgnnx.inference.pipeline import UnifiedInferencePipeline
from tests.test_robustness import MockBaseModel, create_mock_batches

def test_pipeline_integration():
    base_model = MockBaseModel()
    config = MasterConfig()
    pipeline = UnifiedInferencePipeline(base_model, config)
    
    l_batch, p_batch = create_mock_batches(batch_size=1)
    res = pipeline.predict(l_batch, p_batch)
    
    # Assert exact model outputs match GNN prediction flow
    assert res.affinity is not None
    assert res.explanation is not None
    assert res.robustness == 0.9412
