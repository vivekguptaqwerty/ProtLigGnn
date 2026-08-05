import torch
import pytest
from torch_geometric.data import Data, Batch
from models.physics_guided import ProtLigGNNPhysicsGuided, PhysicsGuidedConfig

def create_mock_batch(device):
    # Ligand: 3 nodes, 2 edges, feature dim 78
    x_l = torch.randn(3, 78, device=device)
    edge_l = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long, device=device)
    pos_l = torch.randn(3, 3, device=device)
    ligand = Data(x=x_l, edge_index=edge_l, pos=pos_l)
    
    # Protein: 4 nodes, 3 edges, feature dim 21
    x_p = torch.randn(4, 21, device=device)
    edge_p = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long, device=device)
    pos_p = torch.randn(4, 3, device=device)
    protein = Data(x=x_p, edge_index=edge_p, pos=pos_p)
    
    return Batch.from_data_list([ligand]), Batch.from_data_list([protein])

def test_multitask_forward_cpu_gpu():
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda:0"))
        
    for device in devices:
        model = ProtLigGNNPhysicsGuided(
            ligand_dim=78,
            protein_dim=21,
            hidden_dim=32,
            config=PhysicsGuidedConfig(contact_hidden_dim=32)
        ).to(device)
        
        ligand_batch, protein_batch = create_mock_batch(device)
        
        preds, contact_logits, contact_targets, contexts = model(ligand_batch, protein_batch)
        
        assert preds.shape == (1,)
        assert len(contact_logits) == 1
        assert contact_logits[0].shape == (3, 4)  # 3 ligand atoms, 4 protein residues
        assert contact_targets[0].shape == (3, 4)
        assert len(contexts) == 1
        assert contexts[0].predicted_interaction_embeddings.device == device
