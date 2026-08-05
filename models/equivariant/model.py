import torch
import torch.nn as nn
import inspect
from torch_geometric.data import Batch
from models.foundation.hybrid.model import HybridMultimodalGNN
from models.equivariant.config import EquivariantConfig
from models.equivariant.egnn import EGNN
from models.equivariant.interaction import construct_complex_graph, compute_drift_diagnostics

class EquivariantMultimodalGNN(nn.Module):
    """
    SE(3)-Equivariant Multimodal GNN for protein-ligand binding affinity prediction.
    Augments the promoted Phase 4.3 baseline GNN with an EGNN interaction layer.
    
    Pipeline:
      Foundation Encoders -> Projection -> Cross-Attention Fusion ->
      Geometry Attention -> EGNN Interaction Layer -> Pooling -> Affinity Head
    """
    def __init__(self, config: EquivariantConfig, base_hybrid: HybridMultimodalGNN) -> None:
        super().__init__()
        self.config = config
        self.base_model = base_hybrid  # Wraps HybridMultimodalGNN
        
        # Resolve the GNN base to determine output dimension
        if hasattr(base_hybrid.base_model, "base_model"):
            gnn_model = base_hybrid.base_model.base_model
        else:
            gnn_model = base_hybrid.base_model
        
        # Geometry Attention cross_graph_interaction concatenates [x, context] -> hidden_dim*2
        gnn_output_dim = gnn_model.hidden_dim * 2
        
        # Equivariant interaction block with input projection
        self.egnn = EGNN(config, input_dim=gnn_output_dim)
        
        # Output projection: EGNN hidden_dim back to regressor input dimension
        # The regressor expects hidden_dim * 4 (ligand_pool + protein_pool, each hidden_dim*2)
        # EGNN outputs hidden_dim per node, so we project back to hidden_dim*2
        self.output_projection = nn.Linear(config.hidden_dim, gnn_output_dim)
        
        # Cache for coordinate drift tracking & diagnostics
        self.last_drift_diagnostics = None

    def forward(self, ligand_batch: Batch, protein_batch: Batch) -> torch.Tensor:
        # 1. Resolve geometry attention base model
        hybrid_model = self.base_model
            
        if hasattr(hybrid_model.base_model, "base_model"):
            gnn_model = hybrid_model.base_model.base_model
        else:
            gnn_model = hybrid_model.base_model
            
        # GNN structural representations
        ligand_x = gnn_model.ligand_encoder(ligand_batch.x, ligand_batch.edge_index)
        protein_x = gnn_model.protein_encoder(protein_batch.x, protein_batch.edge_index)
        
        # Project Protein Foundation representations if present
        if hasattr(hybrid_model.base_model, "projection") and hasattr(protein_batch, "foundation_emb"):
            projected_protein = hybrid_model.base_model.projection(protein_batch.foundation_emb)
        else:
            projected_protein = None
            
        # Project Ligand Foundation representations if present
        if hasattr(ligand_batch, "foundation_emb"):
            projected_ligand = hybrid_model.projection_ligand(ligand_batch.foundation_emb)
        else:
            projected_ligand = None
            
        batch_size = ligand_batch.num_graphs
        sig = inspect.signature(gnn_model.cross_graph_interaction)
        has_pos_args = ("ligand_pos" in sig.parameters)
        
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
            
            # Apply Cross-Attention Fusion (Phase 4.3 baseline)
            if p_found_chunk is not None and l_found_chunk is not None:
                if hybrid_model.strategy == "cross_attention":
                    fused_p, fused_l = hybrid_model.cross_attention(p_found_chunk, l_found_chunk)
                else:
                    p_mean = p_found_chunk.mean(dim=0, keepdim=True).expand(l_found_chunk.size(0), -1)
                    l_mean = l_found_chunk.mean(dim=0, keepdim=True).expand(p_found_chunk.size(0), -1)
                    fused_p = hybrid_model.fusion_p(p_found_chunk, l_mean)
                    fused_l = hybrid_model.fusion_l(l_found_chunk, p_mean)
                    
                p_chunk_fused = p_chunk + fused_p
                l_chunk_fused = l_chunk + fused_l
            else:
                p_chunk_fused = p_chunk + (p_found_chunk if p_found_chunk is not None else 0)
                l_chunk_fused = l_chunk + (l_found_chunk if l_found_chunk is not None else 0)
                
            # Apply Geometry Attention GNN (returns hidden_dim*2 per node)
            if has_pos_args:
                l_inter, p_inter = gnn_model.cross_graph_interaction(
                    l_chunk_fused, p_chunk_fused,
                    ligand_pos=ligand_batch.pos[ligand_mask],
                    protein_pos=protein_batch.pos[protein_mask]
                )
            else:
                l_inter, p_inter = gnn_model.cross_graph_interaction(l_chunk_fused, p_chunk_fused)
                
            # Construct unified complex graph for SE(3)-equivariant reasoning
            l_pos = ligand_batch.pos[ligand_mask]
            p_pos = protein_batch.pos[protein_mask]
            
            # Extract local edge indices for this graph in the batch
            l_edge_mask = ligand_batch.batch[ligand_batch.edge_index[0]] == idx
            l_edge_idx = ligand_batch.edge_index[:, l_edge_mask]
            if l_edge_idx.numel() > 0:
                l_offset = ligand_mask.nonzero(as_tuple=True)[0][0]
                l_edge_idx = l_edge_idx - l_offset
                
            p_edge_mask = protein_batch.batch[protein_batch.edge_index[0]] == idx
            p_edge_idx = protein_batch.edge_index[:, p_edge_mask]
            if p_edge_idx.numel() > 0:
                p_offset = protein_mask.nonzero(as_tuple=True)[0][0]
                p_edge_idx = p_edge_idx - p_offset
                
            h_complex, x_complex, edge_index_complex = construct_complex_graph(
                l_inter, l_pos, l_edge_idx,
                p_inter, p_pos, p_edge_idx,
                contact_threshold=8.0
            )
            
            # EGNN Message Passing (projects input_dim -> hidden_dim internally)
            h_out, x_out = self.egnn(h_complex, x_complex, edge_index_complex)
            
            # Monitor coordinate drift
            self.last_drift_diagnostics = compute_drift_diagnostics(x_complex, x_out)
            
            # Project EGNN hidden_dim back to GNN output dim for regressor compatibility
            h_projected = self.output_projection(h_out)
            
            # Split features back to ligand and protein
            N_L = l_inter.size(0)
            l_equivariant = h_projected[:N_L]
            p_equivariant = h_projected[N_L:]
            
            ligand_chunks.append(l_equivariant)
            protein_chunks.append(p_equivariant)
            
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


class EquivariantFoundationOnly(nn.Module):
    """
    Equivariant Foundation Only ablation model.
    Bypasses structural GNN encoders entirely, reasoning directly on projected, 
    fused, and EGNN equivariant message passed foundation representations.
    """
    def __init__(self, config: EquivariantConfig, hybrid_only: nn.Module) -> None:
        super().__init__()
        self.config = config
        self.base_model = hybrid_only  # Wraps HybridFoundationOnly
        self.egnn = EGNN(config, input_dim=config.hidden_dim)
        self.last_drift_diagnostics = None

    def forward(self, ligand_batch: Batch, protein_batch: Batch) -> torch.Tensor:
        # Project Protein & Ligand foundation representations
        if hasattr(protein_batch, "foundation_emb"):
            proj_p = self.base_model.projection_protein(protein_batch.foundation_emb)
        else:
            proj_p = torch.zeros((protein_batch.num_nodes, self.config.hidden_dim), device=ligand_batch.x.device)
            
        if hasattr(ligand_batch, "foundation_emb"):
            proj_l = self.base_model.projection_ligand(ligand_batch.foundation_emb)
        else:
            proj_l = torch.zeros((ligand_batch.num_nodes, self.config.hidden_dim), device=ligand_batch.x.device)
            
        batch_size = ligand_batch.num_graphs
        pools = []
        
        for idx in range(batch_size):
            p_mask = protein_batch.batch == idx
            l_mask = ligand_batch.batch == idx
            
            p_chunk = proj_p[p_mask]
            l_chunk = proj_l[l_mask]
            
            l_pos = ligand_batch.pos[l_mask]
            p_pos = protein_batch.pos[p_mask]
            
            # Apply Cross-Attention Fusion
            if self.base_model.strategy == "cross_attention":
                fused_p, fused_l = self.base_model.cross_attention(p_chunk, l_chunk)
            else:
                p_mean = p_chunk.mean(dim=0, keepdim=True).expand(l_chunk.size(0), -1)
                l_mean = l_chunk.mean(dim=0, keepdim=True).expand(p_chunk.size(0), -1)
                fused_p = self.base_model.fusion_p(p_chunk, l_mean)
                fused_l = self.base_model.fusion_l(l_chunk, p_mean)
                
            # Local chunk edge indices
            l_edge_mask = ligand_batch.batch[ligand_batch.edge_index[0]] == idx
            l_edge_idx = ligand_batch.edge_index[:, l_edge_mask]
            if l_edge_idx.numel() > 0:
                l_offset = l_mask.nonzero(as_tuple=True)[0][0]
                l_edge_idx = l_edge_idx - l_offset
                
            p_edge_mask = protein_batch.batch[protein_batch.edge_index[0]] == idx
            p_edge_idx = protein_batch.edge_index[:, p_edge_mask]
            if p_edge_idx.numel() > 0:
                p_offset = p_mask.nonzero(as_tuple=True)[0][0]
                p_edge_idx = p_edge_idx - p_offset
                
            h_complex, x_complex, edge_index_complex = construct_complex_graph(
                fused_l, l_pos, l_edge_idx,
                fused_p, p_pos, p_edge_idx,
                contact_threshold=8.0
            )
            
            # Message Pass via EGNN
            h_out, x_out = self.egnn(h_complex, x_complex, edge_index_complex)
            self.last_drift_diagnostics = compute_drift_diagnostics(x_complex, x_out)
            
            # Split features back to ligand and protein
            N_L = fused_l.size(0)
            l_pool = h_out[:N_L].mean(dim=0)
            p_pool = h_out[N_L:].mean(dim=0)
            
            combined = torch.cat([p_pool, l_pool], dim=-1)
            pools.append(self.base_model.regressor(combined).squeeze(-1))
            
        return torch.stack(pools, dim=0)
