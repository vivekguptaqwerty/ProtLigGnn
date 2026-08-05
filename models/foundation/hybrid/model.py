import inspect
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from models.foundation.protein.model import ProjectionMLP, HybridProteinGNN
from models.foundation.hybrid.config import HybridFoundationConfig
from models.foundation.hybrid.fusion import (
    ConcatenationFusion,
    WeightedSumFusion,
    GatedFusion,
    ResidualFusion
)
from models.foundation.hybrid.interaction import BidirectionalCrossAttention

class HybridMultimodalGNN(nn.Module):
    """
    Hybrid Multimodal GNN combining protein and ligand representations.
    """
    def __init__(self, config: HybridFoundationConfig, base_hybrid: HybridProteinGNN) -> None:
        super().__init__()
        self.config = config
        self.base_model = base_hybrid  # Wraps HybridProteinGNN
        
        # Determine ligand embedding input dimension based on selected model
        if config.ligand_model == "chemberta":
            self.input_dim_ligand = 384
        elif config.ligand_model == "molformer":
            self.input_dim_ligand = 768
        else:
            self.input_dim_ligand = 256
            
        # Projection Head for ligand foundation representations
        self.projection_ligand = ProjectionMLP(
            input_dim=self.input_dim_ligand,
            output_dim=config.projection_dimension,
            dropout=0.1
        )
        
        # Initialize Fusion strategy
        self.strategy = config.fusion_strategy
        if self.strategy == "concatenation":
            self.fusion_p = ConcatenationFusion(config.projection_dimension)
            self.fusion_l = ConcatenationFusion(config.projection_dimension)
        elif self.strategy == "weighted_sum":
            self.fusion_p = WeightedSumFusion(config.projection_dimension)
            self.fusion_l = WeightedSumFusion(config.projection_dimension)
        elif self.strategy == "gated":
            self.fusion_p = GatedFusion(config.projection_dimension)
            self.fusion_l = GatedFusion(config.projection_dimension)
        elif self.strategy == "residual":
            self.fusion_p = ResidualFusion(config.projection_dimension)
            self.fusion_l = ResidualFusion(config.projection_dimension)
        elif self.strategy == "cross_attention":
            self.cross_attention = BidirectionalCrossAttention(dim=config.projection_dimension)
        elif self.strategy == "no_fusion":
            # Separate head for late ensemble control
            self.affinity_head_lig = nn.Sequential(
                nn.Linear(config.projection_dimension * 2, config.projection_dimension),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(config.projection_dimension, 1)
            )

    def forward(self, ligand_batch: Batch, protein_batch: Batch) -> torch.Tensor:
        # Resolve GNN base encoders (base_model is HybridProteinGNN, its base is ProtLigGNNGeometryAttention)
        if hasattr(self.base_model, "base_model"):
            gnn_model = self.base_model.base_model
        else:
            gnn_model = self.base_model
            
        # GNN structural representations
        ligand_x = gnn_model.ligand_encoder(ligand_batch.x, ligand_batch.edge_index)
        protein_x = gnn_model.protein_encoder(protein_batch.x, protein_batch.edge_index)
        
        # 1. Project Protein Foundation representations if present
        if hasattr(self.base_model, "projection") and hasattr(protein_batch, "foundation_emb"):
            projected_protein = self.base_model.projection(protein_batch.foundation_emb)
        else:
            projected_protein = None
            
        # 2. Project Ligand Foundation representations if present
        if hasattr(ligand_batch, "foundation_emb"):
            projected_ligand = self.projection_ligand(ligand_batch.foundation_emb)
        else:
            projected_ligand = None
            
        ligand_chunks = []
        protein_chunks = []
        batch_size = ligand_batch.num_graphs
        
        # Check signature for backward compatibility
        sig = inspect.signature(gnn_model.cross_graph_interaction)
        has_pos_args = ("ligand_pos" in sig.parameters)
        
        # Handle late ensemble control baseline (no_fusion)
        if self.strategy == "no_fusion":
            # Late ensemble: process protein-only GNN and ligand-only GNN separately and average
            for idx in range(batch_size):
                ligand_mask = ligand_batch.batch == idx
                protein_mask = protein_batch.batch == idx
                
                l_chunk = ligand_x[ligand_mask]
                p_chunk = protein_x[protein_mask]
                
                # Independent additions
                if projected_protein is not None:
                    p_found = projected_protein[protein_mask]
                    if p_found.size(0) != p_chunk.size(0):
                        p_found = p_found.mean(dim=0, keepdim=True).expand(p_chunk.size(0), -1)
                    p_chunk = p_chunk + p_found
                    
                if projected_ligand is not None:
                    l_found = projected_ligand[ligand_mask]
                    if l_found.size(0) != l_chunk.size(0):
                        l_found = l_found.mean(dim=0, keepdim=True).expand(l_chunk.size(0), -1)
                    l_chunk = l_chunk + l_found
                    
                # Cross-graph interaction
                if has_pos_args:
                    l_inter, p_inter = gnn_model.cross_graph_interaction(
                        l_chunk, p_chunk,
                        ligand_pos=ligand_batch.pos[ligand_mask],
                        protein_pos=protein_batch.pos[protein_mask]
                    )
                else:
                    l_inter, p_inter = gnn_model.cross_graph_interaction(l_chunk, p_chunk)
                    
                ligand_chunks.append(l_inter)
                protein_chunks.append(p_inter)
                
            ligand_joint = torch.cat(ligand_chunks, dim=0)
            protein_joint = torch.cat(protein_chunks, dim=0)
            
            # Readout Pooling
            if gnn_model.pooling == "max":
                from torch_geometric.nn import global_max_pool
                ligand_pool = global_max_pool(ligand_joint, ligand_batch.batch)
                protein_pool = global_max_pool(protein_joint, protein_batch.batch)
            else:
                from torch_geometric.nn import global_mean_pool
                ligand_pool = global_mean_pool(ligand_joint, ligand_batch.batch)
                protein_pool = global_mean_pool(protein_joint, protein_batch.batch)
                
            combined = torch.cat([ligand_pool, protein_pool], dim=-1)
            pred_1 = gnn_model.regressor(combined).squeeze(-1)
            pred_2 = self.affinity_head_lig(combined).squeeze(-1)
            return 0.5 * (pred_1 + pred_2)
            
        # Standard fusion pipeline
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
            
            # Apply Fusion
            if p_found_chunk is not None and l_found_chunk is not None:
                if self.strategy == "cross_attention":
                    fused_p, fused_l = self.cross_attention(p_found_chunk, l_found_chunk)
                else:
                    p_mean = p_found_chunk.mean(dim=0, keepdim=True).expand(l_found_chunk.size(0), -1)
                    l_mean = l_found_chunk.mean(dim=0, keepdim=True).expand(p_found_chunk.size(0), -1)
                    fused_p = self.fusion_p(p_found_chunk, l_mean)
                    fused_l = self.fusion_l(l_found_chunk, p_mean)
                    
                p_chunk_fused = p_chunk + fused_p
                l_chunk_fused = l_chunk + fused_l
            else:
                p_chunk_fused = p_chunk + (p_found_chunk if p_found_chunk is not None else 0)
                l_chunk_fused = l_chunk + (l_found_chunk if l_found_chunk is not None else 0)
                
            # Cross-graph interaction
            if has_pos_args:
                l_inter, p_inter = gnn_model.cross_graph_interaction(
                    l_chunk_fused, p_chunk_fused,
                    ligand_pos=ligand_batch.pos[ligand_mask],
                    protein_pos=protein_batch.pos[protein_mask]
                )
            else:
                l_inter, p_inter = gnn_model.cross_graph_interaction(l_chunk_fused, p_chunk_fused)
                
            ligand_chunks.append(l_inter)
            protein_chunks.append(p_inter)
            
        ligand_joint = torch.cat(ligand_chunks, dim=0)
        protein_joint = torch.cat(protein_chunks, dim=0)
        
        # Readout Pooling
        if gnn_model.pooling == "max":
            from torch_geometric.nn import global_max_pool
            ligand_pool = global_max_pool(ligand_joint, ligand_batch.batch)
            protein_pool = global_max_pool(protein_joint, protein_batch.batch)
        else:
            from torch_geometric.nn import global_mean_pool
            ligand_pool = global_mean_pool(ligand_joint, ligand_batch.batch)
            protein_pool = global_mean_pool(protein_joint, protein_batch.batch)
            
        joint = gnn_model.dropout(torch.cat([ligand_pool, protein_pool], dim=-1))
        return gnn_model.regressor(joint).squeeze(-1)


class HybridFoundationOnly(nn.Module):
    """
    Hybrid Foundation Only ablation model.
    Bypasses GNN structure entirely, regressing affinity directly from projected, fused, and pooled foundation embeddings.
    """
    def __init__(self, config: HybridFoundationConfig) -> None:
        super().__init__()
        self.config = config
        
        # Determine protein dimensions
        if config.protein_model == "esm2":
            self.input_dim_protein = 320
        elif config.protein_model == "prot_t5":
            self.input_dim_protein = 1024
        else:
            self.input_dim_protein = 256
            
        # Determine ligand dimensions
        if config.ligand_model == "chemberta":
            self.input_dim_ligand = 384
        elif config.ligand_model == "molformer":
            self.input_dim_ligand = 768
        else:
            self.input_dim_ligand = 256
            
        self.projection_protein = ProjectionMLP(
            input_dim=self.input_dim_protein,
            output_dim=config.projection_dimension,
            dropout=0.1
        )
        self.projection_ligand = ProjectionMLP(
            input_dim=self.input_dim_ligand,
            output_dim=config.projection_dimension,
            dropout=0.1
        )
        
        self.strategy = config.fusion_strategy
        if self.strategy == "concatenation":
            self.fusion_p = ConcatenationFusion(config.projection_dimension)
            self.fusion_l = ConcatenationFusion(config.projection_dimension)
        elif self.strategy == "weighted_sum":
            self.fusion_p = WeightedSumFusion(config.projection_dimension)
            self.fusion_l = WeightedSumFusion(config.projection_dimension)
        elif self.strategy == "gated":
            self.fusion_p = GatedFusion(config.projection_dimension)
            self.fusion_l = GatedFusion(config.projection_dimension)
        elif self.strategy == "residual":
            self.fusion_p = ResidualFusion(config.projection_dimension)
            self.fusion_l = ResidualFusion(config.projection_dimension)
        elif self.strategy == "cross_attention":
            self.cross_attention = BidirectionalCrossAttention(dim=config.projection_dimension)
            
        self.regressor = nn.Sequential(
            nn.Linear(config.projection_dimension * 2, config.projection_dimension),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.projection_dimension, 1)
        )
        
        if self.strategy == "no_fusion":
            self.regressor_lig = nn.Sequential(
                nn.Linear(config.projection_dimension, 1)
            )
            self.regressor_prot = nn.Sequential(
                nn.Linear(config.projection_dimension, 1)
            )

    def forward(self, ligand_batch: Batch, protein_batch: Batch) -> torch.Tensor:
        if hasattr(protein_batch, "foundation_emb"):
            proj_p = self.projection_protein(protein_batch.foundation_emb)
        else:
            proj_p = torch.zeros((protein_batch.num_nodes, self.config.projection_dimension), device=ligand_batch.x.device)
            
        if hasattr(ligand_batch, "foundation_emb"):
            proj_l = self.projection_ligand(ligand_batch.foundation_emb)
        else:
            proj_l = torch.zeros((ligand_batch.num_nodes, self.config.projection_dimension), device=ligand_batch.x.device)
            
        batch_size = ligand_batch.num_graphs
        pools = []
        
        for idx in range(batch_size):
            p_mask = protein_batch.batch == idx
            l_mask = ligand_batch.batch == idx
            
            p_chunk = proj_p[p_mask]
            l_chunk = proj_l[l_mask]
            
            if self.strategy == "no_fusion":
                p_pool = p_chunk.mean(dim=0)
                l_pool = l_chunk.mean(dim=0)
                pred_p = self.regressor_prot(p_pool)
                pred_l = self.regressor_lig(l_pool)
                pools.append(0.5 * (pred_p + pred_l))
            else:
                if self.strategy == "cross_attention":
                    fused_p, fused_l = self.cross_attention(p_chunk, l_chunk)
                else:
                    p_mean = p_chunk.mean(dim=0, keepdim=True).expand(l_chunk.size(0), -1)
                    l_mean = l_chunk.mean(dim=0, keepdim=True).expand(p_chunk.size(0), -1)
                    fused_p = self.fusion_p(p_chunk, l_mean)
                    fused_l = self.fusion_l(l_chunk, p_mean)
                    
                p_pool = fused_p.mean(dim=0)
                l_pool = fused_l.mean(dim=0)
                combined = torch.cat([p_pool, l_pool], dim=-1)
                pools.append(self.regressor(combined).squeeze(-1))
                
        return torch.stack(pools, dim=0)
