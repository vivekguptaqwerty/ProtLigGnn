import torch
from models.geometry_interaction.config import InteractionConfig
from models.geometry_interaction.interfaces import InteractionContext
from models.geometry_interaction.interaction_bias import InteractionBias

def test_interaction_bias_shapes():
    """Verify that interaction bias yields correct attention mask shapes."""
    config = InteractionConfig(
        interaction_dim=16,
        latent_dim=16,
        geometry_dim=16,
        chemistry_dim=16,
        representation_dim=16,
        num_heads=4
    )
    
    # 1. Instantiate module
    bias_module = InteractionBias(config, hidden_dim=64)
    
    # 2. Setup mock InteractionContext
    L, P = 12, 18
    context = InteractionContext(
        version="1.0",
        ligand_pos=torch.randn(L, 3),
        protein_pos=torch.randn(P, 3),
        # 78 features: first 10 for element (e.g. index 1 is Carbon)
        ligand_x_raw=torch.zeros(L, 78),
        protein_x_raw=torch.zeros(P, 30),
        ligand_x_enc=torch.randn(L, 64),
        protein_x_enc=torch.randn(P, 64),
        ligand_batch=torch.zeros(L, dtype=torch.long),
        protein_batch=torch.zeros(P, dtype=torch.long)
    )
    
    # Set Carbon elements
    context.ligand_x_raw[:, 1] = 1.0  # index 1 maps to atomic_choices[1] = 6 (Carbon)
    
    # 3. Forward pass
    bias = bias_module(context)
    
    # Expected output shape: [num_heads, L, P]
    assert bias.shape == (4, L, P)
    assert bias.dtype == torch.float32
    assert bias_module.latest_interaction_emb is not None
    assert bias_module.latest_interaction_emb.shape == (L, P, 16)
