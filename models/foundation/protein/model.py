import inspect
import torch
import torch.nn as nn
from torch_geometric.data import Batch

from models.foundation.projection import ProjectionMLP
from models.foundation.protein.config import ProteinFoundationConfig

class HybridProteinGNN(nn.Module):
    """
    Hybrid Protein-GNN model that integrates Geometry Attention v1.3 GNN
    with pretrained protein foundation representations (ESM-2 / ProtT5).
    
    Fusion occurs at the cross-attention input stage.
    """
    
    def __init__(self, base_gnn: nn.Module, config: ProteinFoundationConfig) -> None:
        super().__init__()
        self.base_model = base_gnn
        self.config = config
        
        # Select input dimension based on selected model
        if config.protein_model == "esm2":
            self.input_dim = 320
        elif config.protein_model == "prot_t5":
            self.input_dim = 1024
        else:
            self.input_dim = 256
            
        # Projection Head: Foundation -> Linear -> LayerNorm -> GELU -> Dropout -> Linear -> Projected
        self.projection = ProjectionMLP(
            input_dim=self.input_dim,
            output_dim=config.projection_dimension,
            dropout=0.1
        )
        
        # Learnable weighted sum parameter if fusion strategy is weighted (added for future-proofing)
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, ligand_batch: Batch, protein_batch: Batch):
        # 1. Base representations from GNN node encoders
        ligand_x = self.base_model.ligand_encoder(ligand_batch.x, ligand_batch.edge_index)
        protein_x = self.base_model.protein_encoder(protein_batch.x, protein_batch.edge_index)
        
        # 2. Extract and project foundation embeddings if active
        if self.config.protein_model != "none" and hasattr(protein_batch, "foundation_emb"):
            found_emb = protein_batch.foundation_emb
            projected_found = self.projection(found_emb)
        else:
            projected_found = None
            
        ligand_chunks = []
        protein_chunks = []
        batch_size = ligand_batch.num_graphs
        
        # Inspect base model signature to handle position-aware models (e.g. Geometry Attention v1.3)
        sig = inspect.signature(self.base_model.cross_graph_interaction)
        has_pos_args = ("ligand_pos" in sig.parameters)
        
        for idx in range(batch_size):
            ligand_mask = ligand_batch.batch == idx
            protein_mask = protein_batch.batch == idx
            
            l_chunk = ligand_x[ligand_mask]
            p_chunk = protein_x[protein_mask]
            
            # Perform Cross-Attention Input Fusion
            if projected_found is not None:
                # Retrieve the projected foundation embedding for this graph
                p_found_chunk = projected_found[protein_mask]
                
                # Check shapes and handle pooling broadcasting if needed
                if p_found_chunk.size(0) != p_chunk.size(0):
                    # Handle Mean / CLS pooling by broadcasting graph-level embedding to all nodes
                    p_found_chunk = p_found_chunk.mean(dim=0, keepdim=True).expand(p_chunk.size(0), -1)
                
                # FUSE: Add them together
                p_chunk_fused = p_chunk + p_found_chunk
            else:
                p_chunk_fused = p_chunk
                
            # Geometry Attention cross-graph interaction
            if has_pos_args:
                l_inter, p_inter = self.base_model.cross_graph_interaction(
                    l_chunk, p_chunk_fused,
                    ligand_pos=ligand_batch.pos[ligand_mask],
                    protein_pos=protein_batch.pos[protein_mask]
                )
            else:
                l_inter, p_inter = self.base_model.cross_graph_interaction(l_chunk, p_chunk_fused)
            
            ligand_chunks.append(l_inter)
            protein_chunks.append(p_inter)
            
        ligand_joint = torch.cat(ligand_chunks, dim=0)
        protein_joint = torch.cat(protein_chunks, dim=0)
        
        # Readout pooling
        if self.base_model.pooling == "max":
            from torch_geometric.nn import global_max_pool
            ligand_pool = global_max_pool(ligand_joint, ligand_batch.batch)
            protein_pool = global_max_pool(protein_joint, protein_batch.batch)
        else:
            from torch_geometric.nn import global_mean_pool
            ligand_pool = global_mean_pool(ligand_joint, ligand_batch.batch)
            protein_pool = global_mean_pool(protein_joint, protein_batch.batch)
            
        joint = self.base_model.dropout(torch.cat([ligand_pool, protein_pool], dim=-1))
        return self.base_model.regressor(joint).squeeze(-1)


class ProteinFoundationOnly(nn.Module):
    """
    Protein Foundation Only ablation model.
    Bypasses GNN structure entirely, regressing affinity directly from projected and pooled foundation embeddings.
    """
    
    def __init__(self, config: ProteinFoundationConfig) -> None:
        super().__init__()
        self.config = config
        
        if config.protein_model == "esm2":
            self.input_dim = 320
        elif config.protein_model == "prot_t5":
            self.input_dim = 1024
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
        if hasattr(protein_batch, "foundation_emb"):
            found_emb = protein_batch.foundation_emb
            projected = self.projection(found_emb)
        else:
            # Empty tensor fallback
            projected = torch.zeros((protein_batch.num_nodes, self.config.projection_dimension), device=ligand_batch.x.device)
            
        # Perform pooling for each complex in the batch
        batch_size = ligand_batch.num_graphs
        pools = []
        
        for idx in range(batch_size):
            protein_mask = protein_batch.batch == idx
            p_proj = projected[protein_mask]
            
            # Pool down to a single representation vector
            p_pool = p_proj.mean(dim=0)
            pools.append(p_pool)
            
        joint = torch.stack(pools, dim=0)
        return self.regressor(joint).squeeze(-1)
