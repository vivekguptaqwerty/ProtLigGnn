import torch
import torch.nn as nn
from typing import List, Tuple
from models.uncertainty.interface import UncertaintyEstimator, PredictionWithUncertainty
from models.uncertainty.config import EnsembleConfig, CalibrationConfig
from models.uncertainty.calibration import calibrate_variance, compute_confidence, compute_prediction_interval

class DeepEnsembleEstimator(UncertaintyEstimator):
    """
    Deep Ensembles Uncertainty Estimator.
    Combines predictions from multiple trained models to compute ensemble mean, 
    predictive variance (epistemic), and ensemble diversity statistics.
    """
    def __init__(
        self,
        ensemble_members: List[nn.Module],
        ensemble_config: EnsembleConfig,
        calib_config: CalibrationConfig
    ) -> None:
        super().__init__()
        self.ensemble_members = nn.ModuleList(ensemble_members)
        self.ensemble_config = ensemble_config
        self.calib_config = calib_config
        
        # Extra tracking attributes for ensemble analysis
        self.last_pairwise_disagreement = 0.0
        self.last_diversity_score = 0.0
        self.last_avg_cosine_similarity = 1.0

    def compute_diversity_metrics(self, predictions_matrix: torch.Tensor) -> None:
        """
        Computes pairwise prediction disagreement, diversity, and cosine similarity.
        predictions_matrix shape: [M, N] where M is ensemble_size, N is batch size.
        """
        M, N = predictions_matrix.shape
        if M <= 1:
            self.last_pairwise_disagreement = 0.0
            self.last_diversity_score = 0.0
            self.last_avg_cosine_similarity = 1.0
            return
            
        # 1. Pairwise prediction disagreement (mean squared difference between members)
        diff_sq_sum = 0.0
        cos_sim_sum = 0.0
        pair_count = 0
        
        for i in range(M):
            for j in range(i + 1, M):
                diff = predictions_matrix[i] - predictions_matrix[j]
                diff_sq_sum += torch.mean(diff ** 2).item()
                
                # Cosine similarity between prediction vectors
                norm_i = torch.linalg.norm(predictions_matrix[i])
                norm_j = torch.linalg.norm(predictions_matrix[j])
                if norm_i > 0 and norm_j > 0:
                    cos_sim = torch.dot(predictions_matrix[i], predictions_matrix[j]) / (norm_i * norm_j)
                    cos_sim_sum += cos_sim.item()
                else:
                    cos_sim_sum += 1.0
                    
                pair_count += 1
                
        self.last_pairwise_disagreement = diff_sq_sum / pair_count
        self.last_avg_cosine_similarity = cos_sim_sum / pair_count
        
        # Diversity score: variance of predictions scaled by mean magnitude
        mean_pred = torch.mean(predictions_matrix, dim=0)
        mean_magnitude = torch.mean(torch.abs(mean_pred)).item()
        var_pred = torch.mean(torch.var(predictions_matrix, dim=0, unbiased=True)).item()
        self.last_diversity_score = var_pred / (mean_magnitude + 1e-6)

    def predict_with_uncertainty(
        self,
        ligand_batch,
        protein_batch
    ) -> PredictionWithUncertainty:
        preds = []
        for member in self.ensemble_members:
            member.eval()
            with torch.no_grad():
                pred = member(ligand_batch, protein_batch)
                preds.append(pred)
                
        # Stack: [M, BatchSize]
        stacked = torch.stack(preds, dim=0)
        
        # Compute mean and variance
        mean_prediction = torch.mean(stacked, dim=0)
        epistemic_variance = torch.var(stacked, dim=0, unbiased=True)
        
        if len(self.ensemble_members) <= 1:
            epistemic_variance = torch.zeros_like(mean_prediction)
            
        # Compute diversity metrics dynamically
        self.compute_diversity_metrics(stacked)
        
        # Homoscedastic aleatoric baseline assumption
        aleatoric_variance = torch.zeros_like(mean_prediction)
        total_variance = aleatoric_variance + epistemic_variance
        
        # Apply calibration
        calibrated = False
        if self.calib_config.calibration_method != "none":
            total_variance = calibrate_variance(total_variance, self.calib_config.variance_multiplier)
            calibrated = True
            
        # Compute confidence score and 95% intervals
        confidence = compute_confidence(total_variance, self.calib_config.temperature)
        lower, upper = compute_prediction_interval(mean_prediction, total_variance, confidence_level=0.95)
        
        return PredictionWithUncertainty(
            affinity=mean_prediction,
            aleatoric_uncertainty=aleatoric_variance,
            epistemic_uncertainty=epistemic_variance,
            total_uncertainty=total_variance,
            confidence=confidence,
            prediction_interval=(lower, upper),
            method="ensemble",
            calibrated=calibrated
        )
