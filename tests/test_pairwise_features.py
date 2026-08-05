import torch
from models.geometry_interaction.interaction_utils import AtomicPropertyEncoder

def test_atomic_property_encoder_dimensions():
    """Verify that property vectors have correct shape and dimension."""
    # Test Carbon (6)
    v_c = AtomicPropertyEncoder.get_properties_vector(6)
    assert isinstance(v_c, torch.Tensor)
    assert v_c.shape == (9,)
    assert v_c.dtype == torch.float32

    # Test unknown element fallback
    v_unk = AtomicPropertyEncoder.get_properties_vector(99)
    assert v_unk.shape == (9,)
    
    # Expected Carbon properties: atomic_num/100, mass/250, electroneg/4
    # 6/100 = 0.06, 12.011/250 = 0.048, 2.55/4 = 0.6375
    assert torch.allclose(v_c[0], torch.tensor(0.06))
    assert torch.allclose(v_c[2], torch.tensor(0.6375))


def test_electronegativity_lookups():
    """Verify that specific electronegativities are mapped correctly."""
    # Fluorine (9) electronegativity is 3.98
    chi_f = AtomicPropertyEncoder.get_electronegativity(9)
    assert chi_f == 3.98

    # Carbon (6) electronegativity is 2.55
    chi_c = AtomicPropertyEncoder.get_electronegativity(6)
    assert chi_c == 2.55
