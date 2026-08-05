import pytest
import torch
from models.geometry_attention.attention_bias import RBFAttentionBias, LearnedAttentionBias, bias_factory

def test_rbf_attention_bias_shapes() -> None:
    """Verifies output shapes and properties of RBFAttentionBias."""
    num_heads = 4
    bias_module = RBFAttentionBias(num_basis=32, start=0.0, stop=12.0, num_heads=num_heads)
    
    # 2 ligand atoms, 3 protein residues
    lig_pos = torch.rand(2, 3) * 5.0
    prot_pos = torch.rand(3, 3) * 5.0
    
    bias = bias_module(lig_pos, prot_pos)
    assert bias.shape == (num_heads, 2, 3)

def test_learned_attention_bias_shapes() -> None:
    """Verifies output shapes of LearnedAttentionBias."""
    num_heads = 4
    bias_module = LearnedAttentionBias(num_heads=num_heads)
    
    lig_pos = torch.rand(2, 3) * 5.0
    prot_pos = torch.rand(3, 3) * 5.0
    
    bias = bias_module(lig_pos, prot_pos)
    assert bias.shape == (num_heads, 2, 3)
    assert torch.all(bias[0] == bias[0, 0, 0]) # Broadcast assertion

def test_bias_factory() -> None:
    """Verifies factory pattern correctly instantiates modules."""
    rbf_bias = bias_factory("rbf", num_basis=16, start=0.0, stop=8.0, num_heads=8)
    assert isinstance(rbf_bias, RBFAttentionBias)
    assert rbf_bias.rbf.num_basis == 16
    
    learned_bias = bias_factory("learned", num_basis=16, start=0.0, stop=8.0, num_heads=8)
    assert isinstance(learned_bias, LearnedAttentionBias)
    
    with pytest.raises(ValueError):
        bias_factory("unknown", num_basis=16, start=0.0, stop=8.0, num_heads=8)
