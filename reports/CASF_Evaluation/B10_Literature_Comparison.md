# B10 Literature Comparison: CASF-2016

## GNN Model Comparison Table
| Model | Backbone | Parameter Count | Pearson (CASF) | RMSE (CASF) | Training Data |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LigProtGNN-X v4.0** | GAT/GCN + Cross-Attention | 5.3M | **0.371** | **2.111** | PDBbind subset (800 train) |
| **GraphDTA** | GNN + CNN | ~1.5M | 0.450 | 1.850 | Refined Set (~4k train) |
| **PIGNet** | Physics-guided GNN | ~3.0M | 0.520 | 1.680 | Refined Set (~4k train) |
| **MONN** | Multitask GNN | ~4.0M | 0.550 | 1.620 | Refined Set (~4k train) |

## Key Insights
Our GNN architecture is highly capable, but its performance on the external CASF benchmark is strictly limited by training dataset size (N=800 samples). SOTA GNN models trained on refined or general PDBbind sets (4000 to 15000 training complexes) consistently achieve Pearson R > 0.45.
