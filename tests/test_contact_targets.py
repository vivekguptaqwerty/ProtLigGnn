import torch
import pytest
from models.physics_guided.interaction_targets import generate_contact_targets

def test_generate_contact_targets_binary():
    pos_lig = torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    pos_prot = torch.tensor([[0.0, 0.0, 3.0], [10.0, 0.0, 0.0]])
    
    # Distance between lig[0] and prot[0] is 3.0 (< 4.0) -> contact
    # Distance between lig[1] and prot[0] is sqrt(25 + 9) = 5.83 (> 4.0) -> no contact
    # Distance between lig[0] and prot[1] is 10.0 (> 4.0) -> no contact
    # Distance between lig[1] and prot[1] is 5.0 (> 4.0) -> no contact
    
    targets = generate_contact_targets(pos_lig, pos_prot, strategy="binary", threshold=4.0)
    expected = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    assert torch.allclose(targets, expected)

def test_generate_contact_targets_other_strategies():
    pos_lig = torch.tensor([[0.0, 0.0, 0.0]])
    pos_prot = torch.tensor([[0.0, 0.0, 3.0]])
    
    # Gaussian
    t_gaussian = generate_contact_targets(pos_lig, pos_prot, strategy="gaussian", gamma=1.0)
    import math
    assert torch.allclose(t_gaussian, torch.tensor([[math.exp(-9.0)]]))
    
    # Distance decay
    t_decay = generate_contact_targets(pos_lig, pos_prot, strategy="distance_decay")
    assert torch.allclose(t_decay, torch.tensor([[1.0 / 10.0]]))
    
    # Inverse distance
    t_inv = generate_contact_targets(pos_lig, pos_prot, strategy="inverse_distance", eps=1e-5)
    assert torch.allclose(t_inv, torch.tensor([[1.0 / (3.0 + 1e-5)]]))
