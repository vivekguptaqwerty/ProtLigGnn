import torch
import pytest
from models.physics_guided.multitask_loss import MultiTaskLoss

def test_multitask_loss_fixed():
    loss_module = MultiTaskLoss(initial_weight=0.2, strategy="fixed")
    affinity_loss = torch.tensor(2.5)
    contact_loss = torch.tensor(1.0)
    
    total, scaled = loss_module(affinity_loss, contact_loss, epoch=5, max_epochs=10)
    
    assert torch.allclose(total, torch.tensor(2.7))
    assert torch.allclose(scaled, torch.tensor(0.2))

def test_multitask_loss_linear_schedule():
    loss_module = MultiTaskLoss(initial_weight=0.1, strategy="linear")
    affinity_loss = torch.tensor(1.0)
    contact_loss = torch.tensor(2.0)
    
    # At epoch 5 / 10, the weight is 0.1 * (1.0 - 0.5) = 0.05
    total, scaled = loss_module(affinity_loss, contact_loss, epoch=5, max_epochs=10)
    
    assert torch.allclose(total, torch.tensor(1.1))
    assert torch.allclose(scaled, torch.tensor(0.1))

def test_multitask_loss_learnable():
    loss_module = MultiTaskLoss(initial_weight=0.1, strategy="learnable")
    affinity_loss = torch.tensor(1.0, requires_grad=True)
    contact_loss = torch.tensor(2.0, requires_grad=True)
    
    total, scaled = loss_module(affinity_loss, contact_loss)
    
    # Backprop should flow to log_lambda parameter
    total.backward()
    assert loss_module.log_lambda.grad is not None
