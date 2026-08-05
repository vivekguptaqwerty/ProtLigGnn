import pytest
import os
from ligprotgnnx.configs.master_config import MasterConfig

def test_master_config_io(tmp_path):
    config = MasterConfig()
    config_file = os.path.join(tmp_path, "test_config.json")
    
    # Export
    config.to_json(config_file)
    assert os.path.exists(config_file)
    
    # Import
    loaded = MasterConfig.from_json(config_file)
    assert loaded.model.hidden_dim == 128
    assert loaded.uncertainty.mc_samples == 30
    assert loaded.explainability.ig_steps == 50
