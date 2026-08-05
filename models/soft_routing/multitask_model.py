import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool, global_max_pool
from typing import Tuple, List, Dict, Any

from models.geometry_attention.geometry_attention_model import ProtLigGNNGeometryAttention
from models.physics_guided.contact_head import ContactPredictionHead
from models.physics_guided.interaction_targets import generate_contact_targets
from models.physics_guided.multitask_loss import MultiTaskLoss

from models.soft_routing.config import RoutingConfig
from models.soft_routing.routing_context import RoutingContext
from models.soft_routing.gated_router import GatedRouter, LinearRouter, ResidualRouter

def get_router_by_name(name: str, config: RoutingConfig) -> nn.Module:
    n = name.lower()
    if n == "linear":
        return LinearRouter(config)
    elif n == "residual":
        return ResidualRouter(config)
    return GatedRouter(config)

class ProtLigGNNSoftRouting(nn.Module):
    """Multitask model wrapping Geometry Attention v1.3 with soft representation routing."""
    
    def __init__(self, ligand_dim: int, protein_dim: int, hidden_dim: int = 128, 
                 no_crossgraph: bool = False, dropout: float = 0.2, 
                 use_residual: bool = True, use_layernorm: bool = True,
                 use_attention: bool = True, pooling: str = "mean",
                 config: RoutingConfig = None) -> None:
        super().__init__()
        self.config = config if config is not None else RoutingConfig()
        
        # Instantiate active v1.3 baseline geometry attention
        from models.geometry_attention.config import AttentionBiasConfig
        attn_config = AttentionBiasConfig(
            num_rbf=32,
            cutoff_distance=12.0,
            learnable_scale=self.config.learnable_temperature,
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
        
        # Instantiate task routers
        # We need independent routers for affinity and contact branches, for ligand and protein embeddings
        router_cfg = RoutingConfig(
            router_type=self.config.router_type,
            routing_level=self.config.routing_level,
            latent_dimension=hidden_dim * 2,
            routing_dimension=hidden_dim * 2,
            num_layers=self.config.num_layers,
            activation=self.config.activation,
            normalization=self.config.normalization,
            dropout=self.config.dropout,
            router_capacity=self.config.router_capacity
        )
        
        self.affinity_router_l = get_router_by_name(self.config.router_type, router_cfg)
        self.affinity_router_p = get_router_by_name(self.config.router_type, router_cfg)
        self.contact_router_l = get_router_by_name(self.config.router_type, router_cfg)
        self.contact_router_p = get_router_by_name(self.config.router_type, router_cfg)
        
        # Reuse contact head from Phase 3.2
        from models.physics_guided.config import PhysicsGuidedConfig
        head_cfg = PhysicsGuidedConfig(
            contact_hidden_dim=self.config.routing_dimension,
            dropout=self.config.dropout,
            activation=self.config.activation,
            normalization=self.config.normalization
        )
        self.contact_head = ContactPredictionHead(
            node_dim=hidden_dim * 2,
            hidden_dim=head_cfg.contact_hidden_dim,
            dropout=head_cfg.dropout,
            activation=head_cfg.activation,
            normalization=head_cfg.normalization
        )
        
        # Multitask loss module
        self.multitask_loss = MultiTaskLoss(
            initial_weight=0.1,
            strategy="fixed"
        )
        
    def forward(self, ligand_batch: Batch, protein_batch: Batch) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[RoutingContext]]:
        # Encoders
        ligand_x = self.base_model.ligand_encoder(ligand_batch.x, ligand_batch.edge_index)
        protein_x = self.base_model.protein_encoder(protein_batch.x, protein_batch.edge_index)
        
        ligand_chunks_aff = []
        protein_chunks_aff = []
        
        contact_logits = []
        contact_targets = []
        contexts = []
        
        batch_size = ligand_batch.num_graphs
        
        for idx in range(batch_size):
            ligand_mask = ligand_batch.batch == idx
            protein_mask = protein_batch.batch == idx
            
            # 1. Geometry Attention cross graph representation (shared embedding)
            lig_chunk_shared, prot_chunk_shared = self.base_model.cross_graph_interaction(
                ligand_x=ligand_x[ligand_mask],
                protein_x=protein_x[protein_mask],
                ligand_pos=ligand_batch.pos[ligand_mask],
                protein_pos=protein_batch.pos[protein_mask]
            )
            
            # 2. Affinity routing
            lig_chunk_aff, gate_l_aff = self.affinity_router_l(lig_chunk_shared)
            prot_chunk_aff, gate_p_aff = self.affinity_router_p(prot_chunk_shared)
            
            # 3. Contact routing
            lig_chunk_contact, gate_l_contact = self.contact_router_l(lig_chunk_shared)
            prot_chunk_contact, gate_p_contact = self.contact_router_p(prot_chunk_shared)
            
            # 4. Contact head predictions
            logits = self.contact_head(lig_chunk_contact, prot_chunk_contact)
            contact_logits.append(logits)
            
            # 5. Targets
            pos_lig = ligand_batch.pos[ligand_mask]
            pos_prot = protein_batch.pos[protein_mask]
            targets = generate_contact_targets(
                pos_lig, pos_prot,
                strategy="binary",
                threshold=4.0
            )
            contact_targets.append(targets)
            
            # Diagnostics & Context
            ctx = RoutingContext(
                shared_embedding=lig_chunk_shared,
                routed_affinity_embedding=lig_chunk_aff,
                routed_contact_embedding=lig_chunk_contact,
                affinity_gates=gate_l_aff,
                contact_gates=gate_l_contact,
                routing_statistics={
                    "gate_l_aff": gate_l_aff,
                    "gate_p_aff": gate_p_aff,
                    "gate_l_contact": gate_l_contact,
                    "gate_p_contact": gate_p_contact
                },
                coordinates={"ligand_pos": pos_lig, "protein_pos": pos_prot},
                predicted_interaction_embeddings=logits
            )
            contexts.append(ctx)
            
            ligand_chunks_aff.append(lig_chunk_aff)
            protein_chunks_aff.append(prot_chunk_aff)
            
        ligand_joint_aff = torch.cat(ligand_chunks_aff, dim=0)
        protein_joint_aff = torch.cat(protein_chunks_aff, dim=0)
        
        # Readout & pooling for affinity
        if self.base_model.pooling == "max":
            ligand_pool = global_max_pool(ligand_joint_aff, ligand_batch.batch)
            protein_pool = global_max_pool(protein_joint_aff, protein_batch.batch)
        elif self.base_model.pooling == "mean_max":
            ligand_pool = torch.cat(
                [
                    global_mean_pool(ligand_joint_aff, ligand_batch.batch),
                    global_max_pool(ligand_joint_aff, ligand_batch.batch),
                ],
                dim=-1,
            )
            protein_pool = torch.cat(
                [
                    global_mean_pool(protein_joint_aff, protein_batch.batch),
                    global_max_pool(protein_joint_aff, protein_batch.batch),
                ],
                dim=-1,
            )
        else:
            ligand_pool = global_mean_pool(ligand_joint_aff, ligand_batch.batch)
            protein_pool = global_mean_pool(protein_joint_aff, protein_batch.batch)
            
        joint = self.base_model.dropout(torch.cat([ligand_pool, protein_pool], dim=-1))
        affinity_preds = self.base_model.regressor(joint).squeeze(-1)
        
        return affinity_preds, contact_logits, contact_targets, contexts

    def compute_regularization_loss(self, contexts: List[RoutingContext]) -> torch.Tensor:
        """Computes gate entropy, sparsity, and diversity regularization losses."""
        reg_loss = torch.tensor(0.0, device=self.base_model.regressor[0].weight.device)
        
        num_gates = 0
        entropy_accum = 0.0
        sparsity_accum = 0.0
        diversity_accum = 0.0
        
        eps = 1e-8
        
        for ctx in contexts:
            # Check gates
            for g_aff, g_contact in [(ctx.affinity_gates, ctx.contact_gates)]:
                if g_aff is not None and g_contact is not None:
                    # Gated routers active
                    num_gates += 1
                    
                    # 1. Entropy: discourage mid-range gate values (close to 0.5)
                    # p log(p) + (1-p) log(1-p)
                    # Minimizing entropy drives gates to 0 or 1
                    p_aff = torch.clamp(g_aff, eps, 1.0 - eps)
                    entropy_aff = - (p_aff * torch.log(p_aff) + (1.0 - p_aff) * torch.log(1.0 - p_aff)).mean()
                    
                    p_contact = torch.clamp(g_contact, eps, 1.0 - eps)
                    entropy_contact = - (p_contact * torch.log(p_contact) + (1.0 - p_contact) * torch.log(1.0 - p_contact)).mean()
                    
                    entropy_accum += (entropy_aff + entropy_contact) / 2.0
                    
                    # 2. Sparsity: L1 regularization on gates
                    sparsity_accum += (g_aff.mean() + g_contact.mean()) / 2.0
                    
                    # 3. Diversity: encourage gates for affinity and contact to be different
                    # Minimize cosine similarity between gate vectors
                    cos_sim = torch.nn.functional.cosine_similarity(g_aff, g_contact, dim=-1).mean()
                    diversity_accum += cos_sim

        if num_gates > 0:
            reg_loss = (
                self.config.entropy_regularization * (entropy_accum / num_gates) +
                self.config.sparsity_regularization * (sparsity_accum / num_gates) +
                self.config.diversity_regularization * (diversity_accum / num_gates)
            )
            
        # Failure detection guards - handle gracefully by zeroing out NaN/Inf components to prevent training crash
        if torch.isnan(reg_loss) or torch.isinf(reg_loss):
            print("Warning: NaN/Inf routing regularization loss detected. Zeroing out to prevent crash.")
            reg_loss = torch.tensor(0.0, device=reg_loss.device, requires_grad=True)
            
        return reg_loss
