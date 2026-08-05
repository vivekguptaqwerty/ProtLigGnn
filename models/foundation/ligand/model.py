import inspect
import torch
import torch.nn as nn
from torch_geometric.data import Batch

from models.foundation.projection import ProjectionMLP
from models.foundation.ligand.config import LigandFoundationConfig

class HybridLigandGNN(nn.Module):
    """
    Hybrid GNN model that integrates ligand molecular foundation representations (ChemBERTa / MolFormer)
    with the Phase 4.1 Protein Foundation baseline.
    
    Fusion occurs at the cross-attention input stage.
    """
    
    def __init__(self, base_hybrid: nn.Module, config: LigandFoundationConfig) -> None:
        super().__init__()
        self.base_model = base_hybrid  # This is the HybridProteinGNN from Phase 4.1
        self.config = config
        
        # Select input dimension based on selected model
        if config.ligand_model == "chemberta":
            self.input_dim = 384
        elif config.ligand_model == "molformer":
            self.input_dim = 768
        else:
            self.input_dim = 256
            
        # Projection Head: Foundation -> Linear -> LayerNorm -> GELU -> Dropout -> Linear -> Projected
        self.projection = ProjectionMLP(
            input_dim=self.input_dim,
            output_dim=config.projection_dimension,
            dropout=0.1
        )

    def forward(self, ligand_batch: Batch, protein_batch: Batch):
        # Resolve GNN base encoders (base_model is HybridProteinGNN, its base is ProtLigGNNGeometryAttention)
        if hasattr(self.base_model, "base_model"):
            gnn_model = self.base_model.base_model
        else:
            gnn_model = self.base_model
            
        ligand_x = gnn_model.ligand_encoder(ligand_batch.x, ligand_batch.edge_index)
        protein_x = gnn_model.protein_encoder(protein_batch.x, protein_batch.edge_index)
        
        # 1. Project Protein Foundation Embeddings if present (Phase 4.1)
        if hasattr(self.base_model, "projection") and hasattr(protein_batch, "foundation_emb"):
            projected_protein = self.base_model.projection(protein_batch.foundation_emb)
        else:
            projected_protein = None
            
        # 2. Project Ligand Foundation Embeddings if present (Phase 4.2)
        if self.config.ligand_model != "none" and hasattr(ligand_batch, "foundation_emb"):
            projected_ligand = self.projection(ligand_batch.foundation_emb)
        else:
            projected_ligand = None
            
        ligand_chunks = []
        protein_chunks = []
        batch_size = ligand_batch.num_graphs
        
        sig = inspect.signature(gnn_model.cross_graph_interaction)
        has_pos_args = ("ligand_pos" in sig.parameters)
        
        for idx in range(batch_size):
            ligand_mask = ligand_batch.batch == idx
            protein_mask = protein_batch.batch == idx
            
            l_chunk = ligand_x[ligand_mask]
            p_chunk = protein_x[protein_mask]
            
            # FUSE protein representation (Phase 4.1)
            if projected_protein is not None:
                p_found_chunk = projected_protein[protein_mask]
                if p_found_chunk.size(0) != p_chunk.size(0):
                    p_found_chunk = p_found_chunk.mean(dim=0, keepdim=True).expand(p_chunk.size(0), -1)
                p_chunk_fused = p_chunk + p_found_chunk
            else:
                p_chunk_fused = p_chunk
                
            # FUSE ligand representation (Phase 4.2)
            if projected_ligand is not None:
                l_found_chunk = projected_ligand[ligand_mask]
                if l_found_chunk.size(0) != l_chunk.size(0):
                    l_found_chunk = l_found_chunk.mean(dim=0, keepdim=True).expand(l_chunk.size(0), -1)
                l_chunk_fused = l_chunk + l_found_chunk
            else:
                l_chunk_fused = l_chunk
                
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


class LigandFoundationOnly(nn.Module):
    """
    Ligand Foundation Only ablation model.
    Bypasses GNN structure entirely, regressing affinity directly from projected and pooled ligand foundation embeddings.
    """
    
    def __init__(self, config: LigandFoundationConfig) -> None:
        super().__init__()
        self.config = config
        
        if config.ligand_model == "chemberta":
            self.input_dim = 384
        elif config.ligand_model == "molformer":
            self.input_dim = 768
        else:
            self.input_dim = 256
            
        self.projection = ProjectionMLP(
            input_dim=self.input_dim,
            output_dim=config.projection_dimension,
            dropout=0.1
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(config.projection_dimension, config.projection_dimension // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.projection_dimension // 2, 1)
        )

    def forward(self, ligand_batch: Batch, protein_batch: Batch):
        if hasattr(ligand_batch, "foundation_emb"):
            found_emb = ligand_batch.foundation_emb
            projected = self.projection(found_emb)
        else:
            projected = torch.zeros((ligand_batch.num_nodes, self.config.projection_dimension), device=ligand_batch.x.device)
            
        batch_size = ligand_batch.num_graphs
        pools = []
        
        for idx in range(batch_size):
            ligand_mask = ligand_batch.batch == idx
            l_proj = projected[ligand_mask]
            
            l_pool = l_proj.mean(dim=0)
            pools.append(l_pool)
            
        joint = torch.stack(pools, dim=0)
        return self.regressor(joint).squeeze(-1)
