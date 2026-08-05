import torch
import torch.nn as nn
from typing import Tuple
from models.uncertainty.interface import UncertaintyEstimator, PredictionWithUncertainty
from models.uncertainty.config import MCDropoutConfig, CalibrationConfig
from models.uncertainty.calibration import calibrate_variance, compute_confidence, compute_prediction_interval

def enable_dropout(m: nn.Module) -> None:
    """
    Recursively sets nn.Dropout modules to train mode to keep dropout active during evaluation.
    """
    if isinstance(m, (nn.Dropout, nn.Dropout2d)):
        m.train()

class MCDropoutEstimator(UncertaintyEstimator):
    """
    Monte Carlo Dropout Uncertainty Estimator.
    Runs multiple forward passes with active dropout during evaluation to sample from
    the posterior weight distribution.
    """
    def __init__(
        self,
        base_model: nn.Module,
        mc_config: MCDropoutConfig,
        calib_config: CalibrationConfig
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.mc_config = mc_config
        self.calib_config = calib_config

    def predict_with_uncertainty(
        self,
        ligand_batch,
        protein_batch
    ) -> PredictionWithUncertainty:
        # Set base model to eval first, then selectively enable only dropout layers
        self.base_model.eval()
        self.base_model.apply(enable_dropout)
        
        samples = []
        mc_samples = self.mc_config.mc_samples
        
        with torch.no_grad():
            for _ in range(mc_samples):
                # Run forward pass
                pred = self.base_model(ligand_batch, protein_batch)
                samples.append(pred)
                
        # Stack samples: [mc_samples, BatchSize]
        stacked = torch.stack(samples, dim=0)
        
        # Calculate mean prediction and epistemic variance
        mean_prediction = torch.mean(stacked, dim=0)
        epistemic_variance = torch.var(stacked, dim=0, unbiased=True)
        
        # Handle single-sample corner case
        if mc_samples <= 1:
            epistemic_variance = torch.zeros_like(mean_prediction)
            
        # Homoscedastic aleatoric baseline assumption (since base model is a point predictor)
        aleatoric_variance = torch.zeros_like(mean_prediction)
        
        total_variance = aleatoric_variance + epistemic_variance
        
        # Apply calibration if configured
        calibrated = False
        if self.calib_config.calibration_method != "none":
            total_variance = calibrate_variance(total_variance, self.calib_config.variance_multiplier)
            calibrated = True
            
        # Compute normalized confidence score and 95% prediction intervals
        confidence = compute_confidence(total_variance, self.calib_config.temperature)
        lower, upper = compute_prediction_interval(mean_prediction, total_variance, confidence_level=0.95)
        
        # Reset model back to clean eval mode
        self.base_model.eval()
        
        return PredictionWithUncertainty(
            affinity=mean_prediction,
            aleatoric_uncertainty=aleatoric_variance,
            epistemic_uncertainty=epistemic_variance,
            total_uncertainty=total_variance,
            confidence=confidence,
            prediction_interval=(lower, upper),
            method="mc_dropout",
            calibrated=calibrated
        )
