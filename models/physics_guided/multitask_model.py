import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool, global_max_pool
from typing import Tuple, List, Dict, Any

from models.geometry_attention.geometry_attention_model import ProtLigGNNGeometryAttention
from models.physics_guided.config import PhysicsGuidedConfig
from models.physics_guided.contact_head import ContactPredictionHead
from models.physics_guided.interaction_targets import generate_contact_targets
from models.physics_guided.context import PhysicsContext
from models.physics_guided.multitask_loss import MultiTaskLoss

class ProtLigGNNPhysicsGuided(nn.Module):
    """Multitask model wrapping Geometry Attention v1.3 with a Contact Prediction Head."""
    
    def __init__(self, ligand_dim: int, protein_dim: int, hidden_dim: int = 128, 
                 no_crossgraph: bool = False, dropout: float = 0.2, 
                 use_residual: bool = True, use_layernorm: bool = True,
                 use_attention: bool = True, pooling: str = "mean",
                 config: PhysicsGuidedConfig = None) -> None:
        super().__init__()
        self.config = config if config is not None else PhysicsGuidedConfig()
        
        # Instantiate active v1.3 baseline geometry attention
        from models.geometry_attention.config import AttentionBiasConfig
        attn_config = AttentionBiasConfig(
            num_rbf=32,
            cutoff_distance=12.0,
            learnable_scale=self.config.learnable_scale,
            initial_alpha=0.1,
            normalize_bias=True,
            num_heads=4
        )
        self.base_model = ProtLigGNNGeometryAttention(
            ligand_dim=ligand_dim,
            protein_dim=protein_dim,
            hidden_dim=hidden_dim,
            no_crossgraph=no_crossgraph,
            dropout=dropout,
            use_residual=use_residual,
            use_layernorm=use_layernorm,
            use_attention=use_attention,
            pooling=pooling,
            config=attn_config
        )
        
        # Add concrete contact prediction head
        self.contact_head = ContactPredictionHead(
            node_dim=hidden_dim * 2,
            hidden_dim=self.config.contact_hidden_dim,
            dropout=self.config.dropout,
            activation=self.config.activation,
            normalization=self.config.normalization
        )
        
        # Add multitask loss module
        self.multitask_loss = MultiTaskLoss(
            initial_weight=self.config.loss_weight,
            strategy=self.config.loss_weight_strategy
        )
        
    def forward(self, ligand_batch: Batch, protein_batch: Batch) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[PhysicsContext]]:
        # Encode via baseline encoders
        ligand_x = self.base_model.ligand_encoder(ligand_batch.x, ligand_batch.edge_index)
        protein_x = self.base_model.protein_encoder(protein_batch.x, protein_batch.edge_index)
        
        ligand_chunks = []
        protein_chunks = []
        batch_size = ligand_batch.num_graphs
        
        contact_logits = []
        contact_targets = []
        contexts = []
        
        for idx in range(batch_size):
            ligand_mask = ligand_batch.batch == idx
            protein_mask = protein_batch.batch == idx
            
            # Form shared interaction representation
            ligand_chunk, protein_chunk = self.base_model.cross_graph_interaction(
                ligand_x=ligand_x[ligand_mask],
                protein_x=protein_x[protein_mask],
                ligand_pos=ligand_batch.pos[ligand_mask],
                protein_pos=protein_batch.pos[protein_mask]
            )
            
            # Predict contacts using the contact head
            logits = self.contact_head(ligand_chunk, protein_chunk)
            contact_logits.append(logits)
            
            # Ground truth targets
            pos_lig = ligand_batch.pos[ligand_mask]
            pos_prot = protein_batch.pos[protein_mask]
            targets = generate_contact_targets(
                pos_lig, pos_prot, 
                strategy="binary", 
                threshold=self.config.contact_threshold
            )
            contact_targets.append(targets)
            
            # Save into PhysicsContext
            ctx = PhysicsContext(
                coordinates={"ligand_pos": pos_lig, "protein_pos": pos_prot},
                node_embeddings={"ligand_chunk": ligand_chunk, "protein_chunk": protein_chunk},
                batch_indices={"idx": torch.tensor(idx, device=ligand_batch.x.device)},
                masks={"ligand_mask": ligand_mask, "protein_mask": protein_mask},
                predicted_interaction_embeddings=logits,
                metadata={"num_ligand_atoms": int(pos_lig.size(0)), "num_protein_residues": int(pos_prot.size(0))}
            )
            contexts.append(ctx)
            
            ligand_chunks.append(ligand_chunk)
            protein_chunks.append(protein_chunk)
            
        ligand_joint = torch.cat(ligand_chunks, dim=0)
        protein_joint = torch.cat(protein_chunks, dim=0)
        
        # Readout & Pooling
        if self.base_model.pooling == "max":
            ligand_pool = global_max_pool(ligand_joint, ligand_batch.batch)
            protein_pool = global_max_pool(protein_joint, protein_batch.batch)
        elif self.base_model.pooling == "mean_max":
            ligand_pool = torch.cat(
                [
                    global_mean_pool(ligand_joint, ligand_batch.batch),
                    global_max_pool(ligand_joint, ligand_batch.batch),
                ],
                dim=-1,
            )
            protein_pool = torch.cat(
                [
                    global_mean_pool(protein_joint, protein_batch.batch),
                    global_max_pool(protein_joint, protein_batch.batch),
                ],
                dim=-1,
            )
        else:
            ligand_pool = global_mean_pool(ligand_joint, ligand_batch.batch)
            protein_pool = global_mean_pool(protein_joint, protein_batch.batch)
            
        joint = self.base_model.dropout(torch.cat([ligand_pool, protein_pool], dim=-1))
        affinity_preds = self.base_model.regressor(joint).squeeze(-1)
        
        return affinity_preds, contact_logits, contact_targets, contexts
