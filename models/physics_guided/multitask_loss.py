import torch
import torch.nn as nn
import math
from typing import Tuple

class MultiTaskLoss(nn.Module):
    """Computes and weights the combined multitask loss for affinity regression and contact prediction."""
    
    def __init__(self, initial_weight: float = 0.1, strategy: str = "fixed") -> None:
        super().__init__()
        self.initial_weight = initial_weight
        self.strategy = strategy
        
        if strategy == "learnable":
            self.log_lambda = nn.Parameter(torch.tensor(math.log(initial_weight)))
            
    def get_weight(self, epoch: int, max_epochs: int) -> float:
        if self.strategy == "fixed":
            return self.initial_weight
        elif self.strategy == "learnable":
            return torch.exp(self.log_lambda).item()
        elif self.strategy == "cosine":
            cos_val = 0.5 * (1.0 + math.cos(math.pi * epoch / max_epochs))
            return self.initial_weight * cos_val
        elif self.strategy == "linear":
            return self.initial_weight * (1.0 - (epoch / max_epochs))
        else:
            return self.initial_weight
            
    def forward(self, affinity_loss: torch.Tensor, contact_loss: torch.Tensor, 
                epoch: int = 0, max_epochs: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.strategy == "learnable":
            # Direct gradient propagation via Parameter
            w = torch.exp(self.log_lambda)
            total_loss = affinity_loss + w * contact_loss
            return total_loss, (w * contact_loss).detach()
        else:
            w = self.get_weight(epoch, max_epochs)
            total_loss = affinity_loss + w * contact_loss
            # Wrap scaled loss in a tensor for uniform return
            scaled_contact = torch.tensor(w * contact_loss.item(), device=contact_loss.device)
            return total_loss, scaled_contact
