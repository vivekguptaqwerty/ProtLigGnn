import pytest
import torch
import tempfile
from pathlib import Path
from models.geometry_attention.config import AttentionBiasConfig
from models.geometry_attention.geometry_attention_model import ProtLigGNNGeometryAttention
from protliggnn_train import save_checkpoint

class MockArgs:
    def __init__(self, model_type="geometry_attention"):
        self.model_type = model_type
        self.attention_bias_type = "rbf"
        self.rbf_basis = 32
        self.rbf_stop = 12.0
        self.attention_learnable_scale = "true"
        self.attention_initial_alpha = 0.15
        self.attention_normalize_bias = "true"

def test_checkpoint_save_and_load() -> None:
    """Verifies that attention configurations and parameters serialize correctly."""
    config = AttentionBiasConfig(
        bias_type="rbf",
        num_rbf=16,
        cutoff_distance=10.0,
        learnable_scale=True,
        initial_alpha=0.15,
        normalize_bias=True,
        num_heads=4
    )
    
    model = ProtLigGNNGeometryAttention(
        ligand_dim=78,
        protein_dim=30,
        hidden_dim=64,
        config=config
    )
    
    # Change log_alpha to verify value restoration
    with torch.no_grad():
        model.cross_attention.log_alpha.fill_(-2.5)
        
    stats = {"epoch": 5, "best_val_rmse": 1.55}
    args = MockArgs()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "checkpoint.pt"
        
        # Save
        save_checkpoint(
            model=model,
            output_path=ckpt_path,
            args=args,
            stats=stats
        )
        
        # Load
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        
        # Verify custom payload additions
        assert "attention_bias_config" in checkpoint
        saved_config = checkpoint["attention_bias_config"]
        assert saved_config["num_rbf"] == 16
        assert saved_config["cutoff_distance"] == 10.0
        assert saved_config["initial_alpha"] == 0.15
        
        # Load into new model
        new_model = ProtLigGNNGeometryAttention(
            ligand_dim=78,
            protein_dim=30,
            hidden_dim=64,
            config=config
        )
        new_model.load_state_dict(checkpoint["model_state_dict"])
        
        # Verify parameter matches the modified log_alpha value
        assert torch.isclose(new_model.cross_attention.log_alpha, torch.tensor(-2.5))
