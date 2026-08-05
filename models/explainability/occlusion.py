import torch
import torch.nn as nn
from typing import Tuple, Any
from models.explainability.interface import ExplanationEngine, ExplanationResult
from models.explainability.config import OcclusionConfig

class OcclusionEngine(ExplanationEngine):
    """
    Computes attributions using systematic node occlusion (masking).
    """
    def __init__(self, base_model: nn.Module, config: OcclusionConfig) -> None:
        super().__init__(base_model)
        self.config = config

    def explain(
        self,
        ligand_batch: Any,
        protein_batch: Any
    ) -> ExplanationResult:
        self.base_model.eval()
        
        # 1. Base prediction
        with torch.no_grad():
            base_pred = self.base_model(ligand_batch, protein_batch)
            
        L = ligand_batch.x.size(0)
        P = protein_batch.x.size(0)
        
        atom_scores = torch.zeros(L, dtype=torch.float32)
        residue_scores = torch.zeros(P, dtype=torch.float32)
        
        # 2. Occlude ligand atoms
        # We loop and mask each atom's features in ligand_batch
        for i in range(L):
            ligand_batch_occ = ligand_batch.clone()
            # Mask features
            ligand_batch_occ.x = ligand_batch.x.clone()
            ligand_batch_occ.x[i] = 0  # Zero out node features
            
            with torch.no_grad():
                occ_pred = self.base_model(ligand_batch_occ, protein_batch)
            
            # Prediction change represents importance
            atom_scores[i] = torch.abs(base_pred - occ_pred).sum()

        # 3. Occlude protein residues
        for j in range(P):
            protein_batch_occ = protein_batch.clone()
            protein_batch_occ.x = protein_batch.x.clone()
            protein_batch_occ.x[j] = 0  # Zero out node features
            
            with torch.no_grad():
                occ_pred = self.base_model(ligand_batch, protein_batch_occ)
                
            residue_scores[j] = torch.abs(base_pred - occ_pred).sum()

        # Normalization
        if atom_scores.max() > 0:
            atom_scores = atom_scores / atom_scores.max()
        if residue_scores.max() > 0:
            residue_scores = residue_scores / residue_scores.max()

        atom_rankings = torch.argsort(atom_scores, descending=True).tolist()
        residue_rankings = torch.argsort(residue_scores, descending=True).tolist()

        interaction_heatmap = torch.outer(residue_scores, atom_scores)

        return ExplanationResult(
            affinity=base_pred,
            confidence=torch.tensor([0.95]),
            residue_scores=residue_scores,
            atom_scores=atom_scores,
            residue_rankings=residue_rankings,
            atom_rankings=atom_rankings,
            interaction_heatmap=interaction_heatmap,
            explanation_method="occlusion",
            explanation_confidence=torch.tensor([1.0])
        )
