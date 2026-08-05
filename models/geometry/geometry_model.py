import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv, global_mean_pool

# Import certified baseline components to guarantee identical implementation
from protliggnn_train import ProteinEncoder, BidirectionalCrossAttention
from .distance_encoding import DistanceEncoder
from .geometry_utils import compute_geometric_features

class GeometryAwareLigandEncoder(nn.Module):
    """Geometry-aware GAT encoder for ligand graphs.
    
    Identical to Baseline v1.1's LigandEncoder but upgraded to accept and process
    explicit geometric edge attributes (edge_attr) during convolutions.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1, 
                 use_layernorm: bool = True, edge_dim: int = 32) -> None:
        """Initializes the GeometryAwareLigandEncoder.
        
        Args:
            input_dim: Dimensionality of input node features (typically 78).
            hidden_dim: Hidden representation dimension (typically 128 or 256).
            dropout: Dropout probability.
            use_layernorm: Whether to apply layer normalization.
            edge_dim: Dimensionality of geometric edge attributes (typically 32).
        """
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim // 4, heads=4, edge_dim=edge_dim, dropout=dropout)
        self.conv2 = GATConv(hidden_dim, hidden_dim // 4, heads=4, edge_dim=edge_dim, dropout=dropout)
        self.conv3 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False, edge_dim=edge_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
        self.norm2 = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
        self.norm3 = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Forward pass of the geometry-aware ligand encoder.
        
        Args:
            x: Node features tensor of shape [num_nodes, input_dim].
            edge_index: Adjacency/connectivity matrix of shape [2, num_edges].
            edge_attr: Geometric edge attributes of shape [num_edges, edge_dim].
            
        Returns:
            Encoded node representation tensor of shape [num_nodes, hidden_dim].
        """
        x = self.norm1(F.elu(self.conv1(x, edge_index, edge_attr=edge_attr)))
        x = self.norm2(F.elu(self.conv2(x, edge_index, edge_attr=edge_attr)))
        x = self.norm3(F.elu(self.conv3(x, edge_index, edge_attr=edge_attr)))
        return x


class ProtLigGNNGeometry(nn.Module):
    """Geometry-Aware ProtLigGNN model for protein-ligand binding affinity prediction.
    
    Integrates explicit Euclidean distance edge features into the ligand encoder, 
    while preserving the baseline configuration for all other modules.
    """
    
    def __init__(self, ligand_dim: int, protein_dim: int, hidden_dim: int = 128, 
                 no_crossgraph: bool = False, dropout: float = 0.2, 
                 use_residual: bool = True, use_layernorm: bool = True,
                 use_attention: bool = True, pooling: str = "mean",
                 num_basis: int = 32, rbf_start: float = 0.0, rbf_stop: float = 12.0,
                 projection_dim: int = 32) -> None:
        """Initializes ProtLigGNNGeometry.
        
        Args:
            ligand_dim: Ligand input node feature dimensions (typically 78).
            protein_dim: Protein input node feature dimensions (typically 30).
            hidden_dim: Hidden dimension size.
            no_crossgraph: If True, bypass cross-graph attention.
            dropout: Dropout probability.
            use_residual: Whether to use residual skip connections.
            use_layernorm: Whether to apply layer normalization.
            use_attention: Whether to use cross-attention projections.
            pooling: Readout pooling strategy ('mean' or 'max').
            num_basis: Number of radial basis functions.
            rbf_start: Start range of distances in Angstroms.
            rbf_stop: End range of distances in Angstroms.
            projection_dim: Dimensionality of projected edge attributes.
        """
        super().__init__()
        self.no_crossgraph = no_crossgraph
        self.use_attention = use_attention
        self.hidden_dim = hidden_dim
        self.pooling = pooling
        
        # 1. Ligand Encoder (Geometry-Aware) and its corresponding distance encoder
        self.distance_encoder = DistanceEncoder(
            num_basis=num_basis,
            start=rbf_start,
            stop=rbf_stop,
            projection_dim=projection_dim
        )
        self.ligand_encoder = GeometryAwareLigandEncoder(
            ligand_dim, hidden_dim, dropout=dropout * 0.5, use_layernorm=use_layernorm, edge_dim=projection_dim
        )
        
        # 2. Protein Encoder (Identical to Baseline v1.1 GCN)
        self.protein_encoder = ProteinEncoder(protein_dim, hidden_dim, use_layernorm=use_layernorm)
        
        # 3. Cross-Attention block (Identical to Baseline v1.1 MHA)
        self.cross_attention = BidirectionalCrossAttention(
            hidden_dim=hidden_dim, 
            num_heads=4, 
            dropout=dropout * 0.5,
            use_residual=use_residual,
            use_layernorm=use_layernorm
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # 4. Regressor MLP Head (Identical to Baseline v1.1)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )

    def cross_graph_interaction(self, ligand_x: torch.Tensor, protein_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cross-graph interaction layer (identical to Baseline v1.1)."""
        if self.no_crossgraph:
            zero_l = torch.zeros_like(ligand_x)
            zero_p = torch.zeros_like(protein_x)
            return torch.cat([ligand_x, zero_l], dim=-1), torch.cat([protein_x, zero_p], dim=-1)

        if not self.use_attention:
            return torch.cat([ligand_x, ligand_x], dim=-1), torch.cat([protein_x, protein_x], dim=-1)

        ligand_context, protein_context = self.cross_attention(ligand_x, protein_x)
        return (
            torch.cat([ligand_x, ligand_context], dim=-1),
            torch.cat([protein_x, protein_context], dim=-1),
        )

    def forward(self, ligand_batch: Batch, protein_batch: Batch) -> torch.Tensor:
        """Forward pass of the Geometry-Aware ProtLigGNN model.
        
        Args:
            ligand_batch: Batch of ligand graphs.
            protein_batch: Batch of protein graphs.
            
        Returns:
            Affinity predictions of shape [batch_size].
        """
        # 1. Compute ligand edge distance representations on the fly
        geom_features = compute_geometric_features(
            pos=ligand_batch.pos,
            edge_index=ligand_batch.edge_index
        )
        edge_attr = self.distance_encoder(geom_features.distances)
        
        # 2. Encode graphs
        ligand_x = self.ligand_encoder(ligand_batch.x, ligand_batch.edge_index, edge_attr=edge_attr)
        protein_x = self.protein_encoder(protein_batch.x, protein_batch.edge_index)

        # 3. Apply baseline equivalent interaction, pooling and regressor logic
        ligand_chunks = []
        protein_chunks = []
        batch_size = ligand_batch.num_graphs
        for idx in range(batch_size):
            ligand_mask = ligand_batch.batch == idx
            protein_mask = protein_batch.batch == idx
            ligand_chunk, protein_chunk = self.cross_graph_interaction(
                ligand_x[ligand_mask],
                protein_x[protein_mask],
            )
            ligand_chunks.append(ligand_chunk)
            protein_chunks.append(protein_chunk)

        ligand_joint = torch.cat(ligand_chunks, dim=0)
        protein_joint = torch.cat(protein_chunks, dim=0)
        
        if self.pooling == "max":
            from torch_geometric.nn import global_max_pool
            ligand_pool = global_max_pool(ligand_joint, ligand_batch.batch)
            protein_pool = global_max_pool(protein_joint, protein_batch.batch)
        else:
            ligand_pool = global_mean_pool(ligand_joint, ligand_batch.batch)
            protein_pool = global_mean_pool(protein_joint, protein_batch.batch)
            
        joint = self.dropout(torch.cat([ligand_pool, protein_pool], dim=-1))
        return self.regressor(joint).squeeze(-1)
