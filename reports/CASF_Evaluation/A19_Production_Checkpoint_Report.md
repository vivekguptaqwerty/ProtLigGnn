# A19 Production Checkpoint Detection: CASF-2016

## Does a production-trained checkpoint exist?
**YES**

## Production Checkpoints Details
| Run ID | Filename | Seed | Max Samples | Epochs Completed | Validation RMSE | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| run_012 | `checkpoint.pt` | 42 | 1000 | 13 | 1.4049 | Production GNN |
| run_013 | `checkpoint.pt` | 123 | 1000 | 18 | 1.3833 | Production GNN |
| run_014 | `checkpoint.pt` | 777 | 1000 | 15 | 1.5503 | Production GNN |
| run_015 | `checkpoint.pt` | 2024 | 1000 | 38 | 1.3837 | Production GNN |
| run_016 | `checkpoint.pt` | 3407 | 1000 | 14 | 1.4851 | Production GNN |

## Explanation
Checkpoints `run_012` through `run_016` represent the fully trained, Phase 6 certified models trained on the maximum available PDBbind training subset (N=1000 samples) across the five designated random seeds.