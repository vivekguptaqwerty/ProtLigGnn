import pytest
import torch
from protliggnn_train import ProtLigGNN, LIGAND_FEATURE_DIM, PROTEIN_FEATURE_DIM

def test_baseline_architecture_and_parameter_count() -> None:
    """Verifies that the certified ProtLigGNN baseline architecture remains unchanged.
    
    Checks shapes, parameter names, and total parameter count to guarantee
    no regressions or structural changes to the baseline.
    """
    model = ProtLigGNN(
        ligand_dim=LIGAND_FEATURE_DIM,
        protein_dim=PROTEIN_FEATURE_DIM,
        hidden_dim=128,  # Canonical baseline dimension
        dropout=0.3181,
        use_residual=True,
        use_layernorm=True,
        use_attention=True,
        pooling="mean"
    )
    
    # 1. Total trainable parameter count check (must be exactly 362,881)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params == 362881, f"Parameter count mismatch: expected 362881, got {total_params}"
    
    # 2. Check individual module existence and shapes to assert structural equivalence
    state_dict = model.state_dict()
    
    expected_shapes = {
        # Ligand Encoder (GATConv with 2 layers of heads=4, hidden=128, and a 3rd layer of heads=1)
        "ligand_encoder.conv1.bias": (128,),
        "ligand_encoder.conv2.bias": (128,),
        "ligand_encoder.conv3.bias": (128,),
        "ligand_encoder.norm1.weight": (128,),
        "ligand_encoder.norm2.weight": (128,),
        "ligand_encoder.norm3.weight": (128,),
        
        # Protein Encoder (GCNConv with 3 layers, hidden=128)
        "protein_encoder.conv1.bias": (128,),
        "protein_encoder.conv2.bias": (128,),
        "protein_encoder.conv3.bias": (128,),
        "protein_encoder.norm1.weight": (128,),
        "protein_encoder.norm2.weight": (128,),
        "protein_encoder.norm3.weight": (128,),
        
        # Regressor MLP (512 -> 256 -> 64 -> 1)
        "regressor.0.weight": (256, 512),
        "regressor.3.weight": (64, 256),
        "regressor.6.weight": (1, 64)
    }
    
    for key, expected_shape in expected_shapes.items():
        assert key in state_dict, f"Expected parameter '{key}' not found in baseline state_dict"
        actual_shape = tuple(state_dict[key].shape)
        assert actual_shape == expected_shape, (
            f"Shape mismatch for parameter '{key}': expected {expected_shape}, got {actual_shape}"
        )


def test_baseline_state_dict_compatibility() -> None:
    """Verifies that model state dict saving and loading works without issues."""
    model_src = ProtLigGNN(
        ligand_dim=LIGAND_FEATURE_DIM,
        protein_dim=PROTEIN_FEATURE_DIM,
        hidden_dim=128
    )
    model_dst = ProtLigGNN(
        ligand_dim=LIGAND_FEATURE_DIM,
        protein_dim=PROTEIN_FEATURE_DIM,
        hidden_dim=128
    )
    
    # Save/load cycle
    state_dict = model_src.state_dict()
    model_dst.load_state_dict(state_dict)
    
    # Verify parameter equivalence
    for p1, p2 in zip(model_src.parameters(), model_dst.parameters()):
        assert torch.equal(p1, p2), "State dict parameters differ after loading"
