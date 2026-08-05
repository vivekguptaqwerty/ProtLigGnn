import torch
from torch_geometric.data import Data, Batch

# Create two dummy Data objects with different node features and foundation embeddings
d1 = Data(x=torch.randn(10, 5), edge_index=torch.tensor([[0], [1]]), foundation_emb=torch.randn(10, 384))
d2 = Data(x=torch.randn(15, 5), edge_index=torch.tensor([[0], [1]]), foundation_emb=torch.randn(15, 384))

batch = Batch.from_data_list([d1, d2])
print("batch.x shape:", batch.x.shape)
print("batch.foundation_emb shape:", batch.foundation_emb.shape)
print("batch.batch shape:", batch.batch.shape)
