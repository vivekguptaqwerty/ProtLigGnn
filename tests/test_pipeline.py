import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from ligprotgnnx.configs.master_config import MasterConfig
from ligprotgnnx.inference.pipeline import PredictionResult, UnifiedInferencePipeline
from tests.test_robustness import MockBaseModel, create_mock_batches

def test_pipeline_execution():
    base_model = MockBaseModel()
    config = MasterConfig()
    pipeline = UnifiedInferencePipeline(base_model, config)
    
    l_batch, p_batch = create_mock_batches(batch_size=2)
    res = pipeline.predict(l_batch, p_batch)
    
    assert isinstance(res, PredictionResult)
    assert res.affinity.shape == (2,)
    assert res.confidence.shape == (2,)
    assert res.prediction_interval[0].shape == (2,)
    assert res.prediction_interval[1].shape == (2,)
    assert res.uncertainty.shape == (2,)
    assert res.metadata["version"] == "4.0.0"
