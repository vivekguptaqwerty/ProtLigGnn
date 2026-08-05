import torch
from pathlib import Path
from models.geometry_interaction.config import InteractionConfig
from models.geometry_interaction.geometry_interaction_model import ProtLigGNNGeometryInteraction

def test_config_dict_serialization():
    """Verify that InteractionConfig dict serialization works bidirectionally."""
    config = InteractionConfig(
        interaction_dim=64,
        geometry_dim=16,
        cache_backend="memory"
    )
    d = config.to_dict()
    assert d["interaction_dim"] == 64
    assert d["geometry_dim"] == 16
    assert d["cache_backend"] == "memory"
    assert d["schema_version"] == "1.0"
    
    config2 = InteractionConfig.from_dict(d)
    assert config2.interaction_dim == 64
    assert config2.geometry_dim == 16
    assert config2.cache_backend == "memory"


def test_checkpoint_save_and_load(tmp_path):
    """Verify that the model state dict and config serialize cleanly to disk."""
    config = InteractionConfig(interaction_dim=16, latent_dim=16)
    model = ProtLigGNNGeometryInteraction(ligand_dim=78, protein_dim=30, hidden_dim=64, config=config)
    
    # Save to temp path
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "attention_bias_config": config.to_dict()
    }, ckpt_path)
    
    # Load and verify
    checkpoint = torch.load(ckpt_path, weights_only=False)
    loaded_config = InteractionConfig.from_dict(checkpoint["attention_bias_config"])
    
    model2 = ProtLigGNNGeometryInteraction(
        ligand_dim=78, protein_dim=30, hidden_dim=64, config=loaded_config
    )
    model2.load_state_dict(checkpoint["model_state_dict"])
    
    # Verify parameter matching
    p1 = list(model.parameters())[0]
    p2 = list(model2.parameters())[0]
    assert torch.allclose(p1, p2)
