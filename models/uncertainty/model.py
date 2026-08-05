import torch
import torch.nn as nn
from typing import List, Optional
from torch_geometric.data import Batch
from models.foundation.hybrid.model import HybridMultimodalGNN
from models.uncertainty.interface import UncertaintyEstimator, PredictionWithUncertainty
from models.uncertainty.config import UncertaintyConfig
from models.uncertainty.mc_dropout import MCDropoutEstimator
from models.uncertainty.ensemble import DeepEnsembleEstimator
from models.uncertainty.evidential import EvidentialRegressionEstimator

class UncertaintyMultimodalGNN(UncertaintyEstimator):
    """
    Uncertainty Estimation Wrapper Model.
    Augments the frozen Phase 4.3 multimodal baseline GNN with calibration and 
    uncertainty routing (MC Dropout, Deep Ensembles, or Evidential Regression).
    """
    def __init__(
        self,
        config: UncertaintyConfig,
        base_hybrid: HybridMultimodalGNN,
        ensemble_checkpoints: Optional[List[str]] = None
    ) -> None:
        super().__init__()
        self.config = config
        self.base_model = base_hybrid
        
        # Initialize the correct estimator based on method config
        if config.method == "mc_dropout":
            self.estimator = MCDropoutEstimator(
                base_model=self.base_model,
                mc_config=config.mc_dropout,
                calib_config=config.calibration
            )
        elif config.method == "ensemble":
            # Deep Ensemble requires a list of loaded model members.
            # If extra ensemble_members aren't provided, we load them or clone the base model
            # for testing and backward compatibility.
            members = [self.base_model]
            
            # Helper to clone base model architecture for ensemble members
            import copy
            ensemble_size = config.ensemble.ensemble_size
            for i in range(1, ensemble_size):
                cloned = copy.deepcopy(self.base_model)
                # If checkpoints are provided, load state dicts
                if ensemble_checkpoints and i < len(ensemble_checkpoints):
                    cloned.load_state_dict(torch.load(ensemble_checkpoints[i], map_location="cpu"))
                members.append(cloned)
                
            self.estimator = DeepEnsembleEstimator(
                ensemble_members=members,
                ensemble_config=config.ensemble,
                calib_config=config.calibration
            )
        elif config.method == "evidential":
            self.estimator = EvidentialRegressionEstimator(
                base_model=self.base_model,
                ev_config=config.evidential,
                calib_config=config.calibration
            )
        else:
            raise ValueError(f"Unknown uncertainty estimation method: {config.method}")

    def forward(self, ligand_batch: Batch, protein_batch: Batch) -> torch.Tensor:
        """
        Standard forward pass fallback for point prediction, keeping the model 
        100% backward compatible with deterministic workflows.
        """
        # Evidential outputs parameter gamma as predicted mean
        if self.config.method == "evidential":
            gamma, _, _, _ = self.estimator.forward_parameters(ligand_batch, protein_batch)
            return gamma
            
        # MC Dropout & Ensemble default back to standard base model point prediction
        return self.base_model(ligand_batch, protein_batch)

    def predict_with_uncertainty(
        self,
        ligand_batch: Batch,
        protein_batch: Batch
    ) -> PredictionWithUncertainty:
        """
        Computes mean predictions, aleatoric/epistemic/total uncertainties,
        and prediction intervals under the configured estimator module.
        """
        return self.estimator.predict_with_uncertainty(ligand_batch, protein_batch)
