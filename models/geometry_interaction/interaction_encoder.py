import torch
import torch.nn as nn
from torch import Tensor
from models.geometry_interaction.interfaces import FeatureEncoder, InteractionContext
from models.geometry.radial_basis import GaussianRadialBasis
from models.geometry_interaction.interaction_utils import compute_pairwise_distances, AtomicPropertyEncoder

class GeometryFeatureEncoder(FeatureEncoder):
    """Geometry encoder using Gaussian RBF expansion of physical distances."""
    def __init__(self, num_basis: int = 32, start: float = 0.0, stop: float = 12.0, geometry_dim: int = 32) -> None:
        super().__init__()
        self.rbf = GaussianRadialBasis(num_basis=num_basis, start=start, stop=stop)
        self.projector = nn.Linear(num_basis, geometry_dim)

    def forward(self, context: InteractionContext) -> Tensor:
        # 1. Compute pairwise distances [L, P]
        dists = compute_pairwise_distances(context.ligand_pos, context.protein_pos)
        # 2. Expand via RBF [L, P, num_basis]
        rbf_features = self.rbf(dists)
        # 3. Project to geometry_dim [L, P, geometry_dim]
        return self.projector(rbf_features)


class ChemistryFeatureEncoder(FeatureEncoder):
    """Chemistry encoder utilizing AtomicPropertyEncoder and raw element features."""
    def __init__(self, chemistry_dim: int = 32) -> None:
        super().__init__()
        # Input features: 9 (from AtomicPropertyEncoder) + 78 (raw ligand) + 30 (raw protein) + 1 (electronegativity diff)
        self.projector = nn.Linear(118, chemistry_dim)
        self.atomic_choices = [5, 6, 7, 8, 9, 15, 16, 17, 35, 53]

    def forward(self, context: InteractionContext) -> Tensor:
        L = context.ligand_x_raw.size(0)
        P = context.protein_x_raw.size(0)
        device = context.ligand_x_raw.device

        # 1. Decode ligand elements and lookup atomic properties
        one_hot_atoms = context.ligand_x_raw[:, :10]
        indices = torch.argmax(one_hot_atoms, dim=-1).tolist()
        
        prop_list = []
        for idx in indices:
            atomic_num = self.atomic_choices[idx]
            prop_list.append(AtomicPropertyEncoder.get_properties_vector(atomic_num))
            
        ligand_props = torch.stack(prop_list, dim=0).to(device)  # [L, 9]
        
        # 2. Compute electronegativity difference
        electronegativities = torch.tensor([
            AtomicPropertyEncoder.get_electronegativity(self.atomic_choices[idx]) for idx in indices
        ], dtype=torch.float32, device=device).unsqueeze(1)  # [L, 1]
        
        # Reference protein amide electronegativity (Nitrogen = 3.04)
        delta_chi = torch.abs(electronegativities - 3.04)  # [L, 1]

        # 3. Form ligand pair input
        ligand_full = torch.cat([ligand_props, context.ligand_x_raw, delta_chi], dim=-1)  # [L, 88]
        
        # 4. Broadcast and concatenate
        lig_expanded = ligand_full.unsqueeze(1).expand(-1, P, -1)  # [L, P, 88]
        prot_expanded = context.protein_x_raw.unsqueeze(0).expand(L, -1, -1)  # [L, P, 30]
        
        chem_pairs = torch.cat([lig_expanded, prot_expanded], dim=-1)  # [L, P, 118]
        
        # 5. Project to chemistry_dim
        return self.projector(chem_pairs)


class RepresentationFeatureEncoder(FeatureEncoder):
    """Representation encoder projecting combined GNN latent features."""
    def __init__(self, hidden_dim: int = 128, representation_dim: int = 32) -> None:
        super().__init__()
        # Input features: x_lig (hidden_dim) + x_prot (hidden_dim) + absolute difference + product
        self.projector = nn.Linear(hidden_dim * 4, representation_dim)
        self.layer_norm = nn.LayerNorm(representation_dim)

    def forward(self, context: InteractionContext) -> Tensor:
        P = context.protein_x_enc.size(0)
        
        # 1. Expand latent embeddings to pair space
        x_lig = context.ligand_x_enc.unsqueeze(1).expand(-1, P, -1)      # [L, P, hidden_dim]
        x_prot = context.protein_x_enc.unsqueeze(0).expand(x_lig.size(0), -1, -1)  # [L, P, hidden_dim]
        
        # 2. Compute difference and product interaction features
        diff_abs = torch.abs(x_lig - x_prot)
        prod = x_lig * x_prot
        
        # 3. Concatenate and project
        rep_features = torch.cat([x_lig, x_prot, diff_abs, prod], dim=-1)  # [L, P, 4 * hidden_dim]
        out = self.projector(rep_features)
        return self.layer_norm(out)
