import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

from .esm2_encoder import ESM2FeatureExtractor
from .physical_features import BiophysicalFeatureEncoder
from .tensor_egnn import TensorEGNN

class SpatialRBFExpansion(nn.Module):
    """Gaussian Radial Basis Function (RBF) Distance Feature Expansion."""
    def __init__(self, num_centers: int = 32, r_min: float = 0.0, r_max: float = 12.0):
        super(SpatialRBFExpansion, self).__init__()
        self.num_centers = num_centers
        self.r_min = r_min
        self.r_max = r_max
        centers = torch.linspace(r_min, r_max, num_centers)
        self.register_buffer('centers', centers)
        self.gamma = 1.0 / ((centers[1] - centers[0]) ** 2)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        # distances: [E, 1] or [E]
        if distances.dim() == 1:
            distances = distances.unsqueeze(-1)
        return torch.exp(-self.gamma * ((distances - self.centers) ** 2))

class DistanceBiasedCrossAttention(nn.Module):
    """Spatial Distance-Biased Multi-Head Cross-Attention with Soft Routing."""
    def __init__(self, hidden_dim: int = 256, num_heads: int = 4, rbf_dim: int = 32):
        super(DistanceBiasedCrossAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.rbf_bias_proj = nn.Linear(rbf_dim, num_heads)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, q_feats: torch.Tensor, k_feats: torch.Tensor, dist_matrix: torch.Tensor, rbf_expansion: SpatialRBFExpansion) -> torch.Tensor:
        # q_feats: [N_q, hidden_dim], k_feats: [N_k, hidden_dim]
        # dist_matrix: [N_q, N_k]
        N_q = q_feats.size(0)
        N_k = k_feats.size(0)
        
        Q = self.q_proj(q_feats).view(N_q, self.num_heads, self.head_dim).transpose(0, 1) # [H, N_q, D]
        K = self.k_proj(k_feats).view(N_k, self.num_heads, self.head_dim).transpose(0, 1) # [H, N_k, D]
        V = self.v_proj(k_feats).view(N_k, self.num_heads, self.head_dim).transpose(0, 1) # [H, N_k, D]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5) # [H, N_q, N_k]

        # Apply RBF spatial distance bias
        rbf_feat = rbf_expansion(dist_matrix.unsqueeze(-1)) # [N_q, N_k, 32]
        rbf_bias = self.rbf_bias_proj(rbf_feat).permute(2, 0, 1) # [H, N_q, N_k]
        scores = scores + rbf_bias

        attn_weights = F.softmax(scores, dim=-1) # [H, N_q, N_k]
        context = torch.matmul(attn_weights, V) # [H, N_q, D]
        context = context.transpose(0, 1).contiguous().view(N_q, self.hidden_dim)
        return self.out_proj(context)

class EvidentialNIGHead(nn.Module):
    """Evidential Normal-Inverse-Gamma (NIG) Regression & Uncertainty Head."""
    def __init__(self, in_dim: int = 512, hidden_dim: int = 256):
        super(EvidentialNIGHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU()
        )
        self.nig_layer = nn.Linear(hidden_dim // 2, 4)

    def forward(self, x: torch.Tensor) -> tuple:
        h = self.mlp(x)
        out = self.nig_layer(h)
        gamma = out[:, 0]
        v = F.softplus(out[:, 1]) + 1e-4
        alpha = F.softplus(out[:, 2]) + 1.0 + 1e-4
        beta = F.softplus(out[:, 3]) + 1e-4
        
        epistemic_var = beta / (v * (alpha - 1.0) + 1e-6)
        return gamma, epistemic_var

class LigProtGNNXv5Model(nn.Module):
    """
    LigProtGNN-X v5.0 Unified SOTA Architecture.
    Combines ESM-2 PLM residue embeddings, Biophysical features, Tensor EGNN layers,
    Distance-Biased Cross-Attention, and Evidential NIG multitask regression head.
    """
    def __init__(self, hidden_dim: int = 256, rbf_dim: int = 32, num_egnn_layers: int = 3):
        super(LigProtGNNXv5Model, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Feature Encoders
        self.esm2_extractor = ESM2FeatureExtractor(embed_dim=1280)
        self.esm2_proj = nn.Sequential(
            nn.Linear(1280, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.protein_node_proj = nn.Sequential(
            nn.Linear(20, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.ligand_node_proj = nn.Sequential(
            nn.Linear(35, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.bio_encoder = BiophysicalFeatureEncoder(in_dim=8, out_dim=hidden_dim)
        
        # Spatial RBF Expansion
        self.rbf_expansion = SpatialRBFExpansion(num_centers=rbf_dim)
        
        # GNN Backbone Encoders
        self.protein_gcn = GCNConv(hidden_dim, hidden_dim)
        self.ligand_gat = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        
        # Equivariant Tensor EGNN
        self.tensor_egnn = TensorEGNN(node_dim=hidden_dim, edge_dim=rbf_dim, num_layers=num_egnn_layers)
        
        # Geometric Cross-Attention
        self.cross_attn = DistanceBiasedCrossAttention(hidden_dim=hidden_dim, rbf_dim=rbf_dim)
        
        # Regressor Head
        self.evidential_head = EvidentialNIGHead(in_dim=hidden_dim * 2, hidden_dim=hidden_dim)

    def forward(self, data) -> tuple:
        """
        Expects PyG Batch object containing protein and ligand node attributes, spatial positions, and edge connectivity.
        """
        # Handle protein stream
        p_x = getattr(data, 'x_protein', None)
        p_pos = getattr(data, 'pos_protein', None)
        p_edge_index = getattr(data, 'edge_index_protein', None)
        
        if p_x is None:
            # Fallback tensor creation
            p_x = torch.zeros(16, 20, device=data.x.device if hasattr(data, 'x') else 'cpu')
            p_pos = torch.zeros(16, 3, device=p_x.device)
            p_edge_index = torch.zeros(2, 16, dtype=torch.long, device=p_x.device)
            
        p_h_base = self.protein_node_proj(p_x[:, :20])
        p_h_esm = self.esm2_proj(self.esm2_extractor(p_x[:, :20]))
        p_h = p_h_base + p_h_esm

        # Handle ligand stream
        l_x = getattr(data, 'x_ligand', None)
        l_pos = getattr(data, 'pos_ligand', None)
        l_edge_index = getattr(data, 'edge_index_ligand', None)
        
        if l_x is None:
            l_x = torch.zeros(16, 35, device=p_x.device)
            l_pos = torch.zeros(16, 3, device=p_x.device)
            l_edge_index = torch.zeros(2, 16, dtype=torch.long, device=p_x.device)
            
        l_h = self.ligand_node_proj(l_x[:, :35])

        # Execute GNN Encoders
        p_h = F.silu(self.protein_gcn(p_h, p_edge_index))
        l_h = F.silu(self.ligand_gat(l_h, l_edge_index))

        # Compute RBF edge features for protein graph
        p_row, p_col = p_edge_index[0], p_edge_index[1]
        p_dists = torch.norm(p_pos[p_row] - p_pos[p_col], dim=-1, keepdim=True)
        p_rbf = self.rbf_expansion(p_dists)

        # Tensor EGNN Layers
        p_h, p_pos = self.tensor_egnn(p_h, p_pos, p_edge_index, p_rbf)

        # Inter-molecular Cross-Attention Fusion
        dist_matrix = torch.cdist(p_pos, l_pos) # [N_p, N_l]
        fused_p_h = self.cross_attn(p_h, l_h, dist_matrix, self.rbf_expansion)

        # Global Attention Pooling
        p_batch = getattr(data, 'x_protein_batch', torch.zeros(p_h.size(0), dtype=torch.long, device=p_h.device))
        l_batch = getattr(data, 'x_ligand_batch', torch.zeros(l_h.size(0), dtype=torch.long, device=l_h.device))

        z_pocket = global_mean_pool(fused_p_h, p_batch)
        z_ligand = global_mean_pool(l_h, l_batch)

        z_joint = torch.cat([z_pocket, z_ligand], dim=-1)

        # Evidential Output Head
        pred_affinity, epistemic_var = self.evidential_head(z_joint)
        return pred_affinity, epistemic_var
