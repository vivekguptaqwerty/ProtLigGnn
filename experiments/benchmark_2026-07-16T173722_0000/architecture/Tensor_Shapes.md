# Tensor Shapes Report
| Layer | Input Shape | Operation | Output Shape |
| :--- | :---: | :---: | :---: |
| GNN Node Enc | (N, 78) | Message Passing | (N, 256) |
| Cross-Attn Query | (L, 256) | Linear Projection | (L, 256) |
| Readout pooling | (L, 256) | Global Mean Pool | (256,) |
| MLP Regressor | (512,) | Concatenated Linear | (1,) |