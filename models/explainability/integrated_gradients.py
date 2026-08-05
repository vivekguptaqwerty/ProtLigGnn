import torch
import torch.nn as nn
from typing import Tuple, Any, List
from models.explainability.interface import ExplanationEngine, ExplanationResult
from models.explainability.config import IntegratedGradientsConfig
from models.explainability.residue_importance import map_atoms_to_residues

class IntegratedGradientsEngine(ExplanationEngine):
    """
    Computes Integrated Gradients for node representation features.
    """
    def __init__(self, base_model: nn.Module, config: IntegratedGradientsConfig) -> None:
        super().__init__(base_model)
        self.config = config

    def explain(
        self,
        ligand_batch: Any,
        protein_batch: Any
    ) -> ExplanationResult:
        self.base_model.eval()
        
        # Get baseline (zero representation)
        steps = self.config.steps
        
        # We perform path integration w.r.t the encoder outputs (representations)
        if hasattr(self.base_model, "base_model"):
            gnn_model = self.base_model.base_model
            if hasattr(gnn_model, "base_model"):
                gnn_model = gnn_model.base_model
        else:
            gnn_model = self.base_model

        # Baseline & input representations
        with torch.no_grad():
            ligand_x = gnn_model.ligand_encoder(ligand_batch.x, ligand_batch.edge_index)
            protein_x = gnn_model.protein_encoder(protein_batch.x, protein_batch.edge_index)
            
            # Predict baseline (with zero representations)
            ligand_x_base = torch.zeros_like(ligand_x)
            protein_x_base = torch.zeros_like(protein_x)
            
            # Run baseline prediction
            l_inter_base, p_inter_base = gnn_model.cross_graph_interaction(
                ligand_x_base, protein_x_base, ligand_batch.pos, protein_batch.pos
            )
            from torch_geometric.nn import global_mean_pool
            lig_pool_base = global_mean_pool(l_inter_base, ligand_batch.batch)
            prot_pool_base = global_mean_pool(p_inter_base, protein_batch.batch)
            joint_base = gnn_model.dropout(torch.cat([lig_pool_base, prot_pool_base], dim=-1))
            pred_base = gnn_model.regressor(joint_base).squeeze(-1)
            
            # Predict actual
            l_inter, p_inter = gnn_model.cross_graph_interaction(
                ligand_x, protein_x, ligand_batch.pos, protein_batch.pos
            )
            lig_pool = global_mean_pool(l_inter, ligand_batch.batch)
            prot_pool = global_mean_pool(p_inter, protein_batch.batch)
            joint = gnn_model.dropout(torch.cat([lig_pool, prot_pool], dim=-1))
            pred_actual = gnn_model.regressor(joint).squeeze(-1)
            
            target_diff = pred_actual - pred_base

        # Calculate path gradients
        accumulated_ligand_grad = torch.zeros_like(ligand_x)
        accumulated_protein_grad = torch.zeros_like(protein_x)
        
        for step in range(1, steps + 1):
            alpha = step / steps
            
            # Interpolated features
            ligand_step = ligand_x_base + alpha * (ligand_x - ligand_x_base)
            protein_step = protein_x_base + alpha * (protein_x - protein_x_base)
            
            ligand_step.requires_grad_(True)
            protein_step.requires_grad_(True)
            
            # Predict
            l_inter_s, p_inter_s = gnn_model.cross_graph_interaction(
                ligand_step, protein_step, ligand_batch.pos, protein_batch.pos
            )
            l_pool_s = global_mean_pool(l_inter_s, ligand_batch.batch)
            p_pool_s = global_mean_pool(p_inter_s, protein_batch.batch)
            joint_s = gnn_model.dropout(torch.cat([l_pool_s, p_pool_s], dim=-1))
            pred_s = gnn_model.regressor(joint_s).squeeze(-1)
            
            grad_outputs = torch.ones_like(pred_s)
            grads = torch.autograd.grad(
                outputs=pred_s,
                inputs=[ligand_step, protein_step],
                grad_outputs=grad_outputs,
                create_graph=False,
                retain_graph=False
            )
            
            accumulated_ligand_grad += grads[0]
            accumulated_protein_grad += grads[1]

        # Calculate integrated gradients: (input - baseline) * average_gradient
        ig_ligand = (ligand_x - ligand_x_base) * (accumulated_ligand_grad / steps)
        ig_protein = (protein_x - protein_x_base) * (accumulated_protein_grad / steps)
        
        # Verify completeness axiom: sum(IG) should be close to pred_actual - pred_base
        ig_sum_lig = ig_ligand.sum()
        ig_sum_prot = ig_protein.sum()
        total_ig_sum = ig_sum_lig + ig_sum_prot
        
        completeness_error = torch.abs(total_ig_sum - target_diff.sum())
        
        # Compute attribution scores by taking norm over the feature dimension
        atom_scores = torch.norm(ig_ligand, p=2, dim=-1)
        residue_scores = torch.norm(ig_protein, p=2, dim=-1)
        
        # Max-normalize for visual heatmap readability
        if atom_scores.max() > 0:
            atom_scores = atom_scores / atom_scores.max()
        if residue_scores.max() > 0:
            residue_scores = residue_scores / residue_scores.max()
            
        atom_rankings = torch.argsort(atom_scores, descending=True).tolist()
        residue_rankings = torch.argsort(residue_scores, descending=True).tolist()
        
        # Compute a simple pairwise interaction heatmap as outer product of attributions
        interaction_heatmap = torch.outer(residue_scores, atom_scores)
        
        return ExplanationResult(
            affinity=pred_actual,
            confidence=torch.tensor([0.95]),  # Calibrated dummy placeholder
            residue_scores=residue_scores,
            atom_scores=atom_scores,
            residue_rankings=residue_rankings,
            atom_rankings=atom_rankings,
            interaction_heatmap=interaction_heatmap,
            explanation_method="integrated_gradients",
            explanation_confidence=torch.tensor([1.0 - min(1.0, completeness_error.item())])
        )
