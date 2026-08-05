import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from models.uncertainty.interface import UncertaintyEstimator, PredictionWithUncertainty
from models.uncertainty.config import EvidentialConfig, CalibrationConfig
from models.uncertainty.calibration import calibrate_variance, compute_confidence, compute_prediction_interval

class EvidentialRegressionHead(nn.Module):
    """
    Normal-Inverse-Gamma Evidential Regression Head.
    Outputs NIG parameters (gamma, v, alpha, beta) to estimate both aleatoric and epistemic uncertainty.
    """
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.dense = nn.Linear(input_dim, 4)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.dense(x)
        gamma, log_v, log_alpha, log_beta = torch.chunk(output, 4, dim=-1)
        
        # Apply activation constraints to keep parameters in valid ranges
        # v > 0, alpha > 1, beta > 0
        v = F.softplus(log_v) + 1e-6
        alpha = F.softplus(log_alpha) + 1.0 + 1e-6
        beta = F.softplus(log_beta) + 1e-6
        
        return gamma.squeeze(-1), v.squeeze(-1), alpha.squeeze(-1), beta.squeeze(-1)


def evidential_regression_loss(
    gamma: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    target: torch.Tensor,
    lambda_val: float = 0.1
) -> torch.Tensor:
    """
    Computes the Negative Log-Likelihood of the NIG distribution + Evidential regularization.
    """
    # 1. Negative Log-Likelihood (NLL) of NIG
    # Standard formula:
    # log_likelihood = 0.5 * log(pi/v) - alpha * log(beta) + lgamma(alpha) + (alpha + 0.5) * log(beta + v*(target - gamma)^2 / (2*(v+1))) - lgamma(alpha + 0.5)
    # Stably computed:
    v = torch.clamp(v, min=1e-5)
    alpha = torch.clamp(alpha, min=1.00001)
    beta = torch.clamp(beta, min=1e-5)
    
    t1 = 0.5 * torch.log(torch.pi / v)
    t2 = -alpha * torch.log(beta)
    t3 = torch.lgamma(alpha)
    
    error_sq = (target - gamma) ** 2
    log_arg = beta + (v * error_sq) / (2.0 * (v + 1.0))
    t4 = (alpha + 0.5) * torch.log(log_arg)
    t5 = -torch.lgamma(alpha + 0.5)
    
    nll = t1 + t2 + t3 + t4 + t5
    nll_loss = torch.mean(nll)
    
    # 2. Regularization loss to penalize high uncertainty on high errors
    # regularizer = |target - gamma| * (2 * v + alpha)
    reg = torch.abs(target - gamma) * (2.0 * v + alpha)
    reg_loss = torch.mean(reg)
    
    return nll_loss + lambda_val * reg_loss


class EvidentialRegressionEstimator(UncertaintyEstimator):
    """
    Evidential Regression Uncertainty Estimator.
    Uses Normal-Inverse-Gamma outputs to estimate aleatoric, epistemic, and total variance in a single forward pass.
    """
    def __init__(
        self,
        base_model: nn.Module,
        ev_config: EvidentialConfig,
        calib_config: CalibrationConfig
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.ev_config = ev_config
        self.calib_config = calib_config
        
        # Replace the final linear projection in the regressor with the Evidential head
        # The base hybrid model regressor has:
        # self.regressor = nn.Sequential(
        #     nn.Linear(hidden_dim * 4, hidden_dim * 2),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim * 2, hidden_dim // 2),
        #     nn.ReLU(),
        #     nn.Dropout(dropout * 0.5),
        #     nn.Linear(hidden_dim // 2, 1),
        # )
        
        # We extract target dimension from the last Linear layer in regressor sequence
        regressor = base_model.regressor if hasattr(base_model, "regressor") else base_model.base_model.regressor
        input_features = regressor[-1].in_features
        
        # Replacing the last layer in-place
        self.evidential_head = EvidentialRegressionHead(input_features)
        
        # Save old regressor last layer for backward compatibility
        self.old_last_layer = regressor[-1]
        
        # Define a proxy regression model wrapper that uses evidential head
        # We will override base_model's last regression layer dynamically during forward if needed,
        # or call it manually inside our custom predict_with_uncertainty logic.

    def forward_parameters(self, ligand_batch, protein_batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes GNN forward pass up to the final layer, then applies evidential head.
        """
        # Resolve GNN base models
        hybrid_model = self.base_model
        if hasattr(hybrid_model.base_model, "base_model"):
            gnn_model = hybrid_model.base_model.base_model
        else:
            gnn_model = hybrid_model.base_model
            
        # Standard GNN encoders + fusion forward passes
        ligand_x = gnn_model.ligand_encoder(ligand_batch.x, ligand_batch.edge_index)
        protein_x = gnn_model.protein_encoder(protein_batch.x, protein_batch.edge_index)
        
        if hasattr(hybrid_model.base_model, "projection") and hasattr(protein_batch, "foundation_emb"):
            projected_protein = hybrid_model.base_model.projection(protein_batch.foundation_emb)
        else:
            projected_protein = None
            
        if hasattr(ligand_batch, "foundation_emb"):
            projected_ligand = hybrid_model.projection_ligand(ligand_batch.foundation_emb)
        else:
            projected_ligand = None
            
        batch_size = ligand_batch.num_graphs
        ligand_chunks = []
        protein_chunks = []
        
        for idx in range(batch_size):
            ligand_mask = ligand_batch.batch == idx
            protein_mask = protein_batch.batch == idx
            
            l_chunk = ligand_x[ligand_mask]
            p_chunk = protein_x[protein_mask]
            
            p_found_chunk = None
            l_found_chunk = None
            
            if projected_protein is not None:
                p_found_chunk = projected_protein[protein_mask]
                if p_found_chunk.size(0) != p_chunk.size(0):
                    p_found_chunk = p_found_chunk.mean(dim=0, keepdim=True).expand(p_chunk.size(0), -1)
                    
            if projected_ligand is not None:
                l_found_chunk = projected_ligand[ligand_mask]
                if l_found_chunk.size(0) != l_chunk.size(0):
                    l_found_chunk = l_found_chunk.mean(dim=0, keepdim=True).expand(l_chunk.size(0), -1)
            
            if p_found_chunk is not None and l_found_chunk is not None:
                if hybrid_model.strategy == "cross_attention":
                    fused_p, fused_l = hybrid_model.cross_attention(p_found_chunk, l_found_chunk)
                else:
                    p_mean = p_found_chunk.mean(dim=0, keepdim=True).expand(l_found_chunk.size(0), -1)
                    l_mean = l_found_chunk.mean(dim=0, keepdim=True).expand(p_found_chunk.size(0), -1)
                    fused_p = hybrid_model.fusion_p(p_found_chunk, l_mean)
                    fused_l = hybrid_model.fusion_l(l_found_chunk, p_mean)
                p_chunk = p_chunk + fused_p
                l_chunk = l_chunk + fused_l
            else:
                p_chunk = p_chunk + (p_found_chunk if p_found_chunk is not None else 0)
                l_chunk = l_chunk + (l_found_chunk if l_found_chunk is not None else 0)
                
            l_inter, p_inter = gnn_model.cross_graph_interaction(
                l_chunk, p_chunk,
                ligand_pos=ligand_batch.pos[ligand_mask],
                protein_pos=protein_batch.pos[protein_mask]
            )
            ligand_chunks.append(l_inter)
            protein_chunks.append(p_inter)
            
        ligand_joint = torch.cat(ligand_chunks, dim=0)
        protein_joint = torch.cat(protein_chunks, dim=0)
        
        # Readout Pooling
        from torch_geometric.nn import global_mean_pool, global_max_pool
        if gnn_model.pooling == "max":
            ligand_pool = global_max_pool(ligand_joint, ligand_batch.batch)
            protein_pool = global_max_pool(protein_joint, protein_batch.batch)
        else:
            ligand_pool = global_mean_pool(ligand_joint, ligand_batch.batch)
            protein_pool = global_mean_pool(protein_joint, protein_batch.batch)
            
        joint = gnn_model.dropout(torch.cat([ligand_pool, protein_pool], dim=-1))
        
        # Forward pass through regressor MLP up to the last layer
        regressor_hidden = joint
        for layer in gnn_model.regressor[:-1]:
            regressor_hidden = layer(regressor_hidden)
            
        # Apply evidential regression head
        return self.evidential_head(regressor_hidden)

    def predict_with_uncertainty(
        self,
        ligand_batch,
        protein_batch
    ) -> PredictionWithUncertainty:
        gamma, v, alpha, beta = self.forward_parameters(ligand_batch, protein_batch)
        
        # Calculate uncertainties
        # Aleatoric uncertainty: beta / (alpha - 1)
        aleatoric_variance = beta / (alpha - 1.0)
        # Epistemic uncertainty: beta / (v * (alpha - 1))
        epistemic_variance = beta / (v * (alpha - 1.0))
        
        total_variance = aleatoric_variance + epistemic_variance
        
        # Apply calibration
        calibrated = False
        if self.calib_config.calibration_method != "none":
            total_variance = calibrate_variance(total_variance, self.calib_config.variance_multiplier)
            calibrated = True
            
        # Compute confidence score and 95% intervals
        confidence = compute_confidence(total_variance, self.calib_config.temperature)
        lower, upper = compute_prediction_interval(gamma, total_variance, confidence_level=0.95)
        
        return PredictionWithUncertainty(
            affinity=gamma,
            aleatoric_uncertainty=aleatoric_variance,
            epistemic_uncertainty=epistemic_variance,
            total_uncertainty=total_variance,
            confidence=confidence,
            prediction_interval=(lower, upper),
            method="evidential",
            calibrated=calibrated
        )
