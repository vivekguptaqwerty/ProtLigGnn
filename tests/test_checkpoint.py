import torch
import pytest
from pathlib import Path
from models.physics_guided import ProtLigGNNPhysicsGuided, PhysicsGuidedConfig

def test_config_serialization():
    config = PhysicsGuidedConfig(
        contact_threshold=4.5,
        loss_weight=0.15,
        loss_weight_strategy="linear",
        contact_hidden_dim=64
    )
    
    d = config.to_dict()
    restored = PhysicsGuidedConfig.from_dict(d)
    
    assert restored.contact_threshold == 4.5
    assert restored.loss_weight == 0.15
    assert restored.loss_weight_strategy == "linear"
    assert restored.contact_hidden_dim == 64

def test_model_state_dict():
    model = ProtLigGNNPhysicsGuided(
        ligand_dim=78,
        protein_dim=21,
        hidden_dim=32,
        config=PhysicsGuidedConfig(contact_hidden_dim=32)
    )
    
    state_dict = model.state_dict()
    assert "contact_head.mlp.0.weight" in state_dict
    assert "base_model.regressor.0.weight" in state_dict
