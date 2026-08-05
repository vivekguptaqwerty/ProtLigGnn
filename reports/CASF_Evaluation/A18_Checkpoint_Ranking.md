# A18 Checkpoint Ranking: CASF-2016

This report ranks and categorizes the checkpoint files by scientific purpose (Smoke Test, Ablation, HPO, Production).

## Checkpoint Classification Table
| Path | Max Samples | Epochs | Purpose | Explanation |
| :--- | :--- | :--- | :--- | :--- |
| `experiments/ablation_no_attention/checkpoint.pt` | 1000 | 20 | **Ablation** | Ablation study checkpoint evaluating GNN subcomponents. |
| `experiments/ablation_no_attention_seed_123/checkpoint.pt` | 1000 | 16 | **Ablation** | Ablation study checkpoint evaluating GNN subcomponents. |
| `experiments/ablation_no_attention_seed_2024/checkpoint.pt` | 1000 | 18 | **Ablation** | Ablation study checkpoint evaluating GNN subcomponents. |
| `experiments/ablation_no_attention_seed_3407/checkpoint.pt` | 1000 | 20 | **Ablation** | Ablation study checkpoint evaluating GNN subcomponents. |
| `experiments/ablation_no_attention_seed_42/checkpoint.pt` | 1000 | 15 | **Ablation** | Ablation study checkpoint evaluating GNN subcomponents. |
| `experiments/ablation_no_attention_seed_777/checkpoint.pt` | 1000 | 18 | **Ablation** | Ablation study checkpoint evaluating GNN subcomponents. |
| `experiments/ablation_no_crossgraph/checkpoint.pt` | 1000 | 20 | **Ablation** | Ablation study checkpoint evaluating GNN subcomponents. |
| `experiments/ablation_no_crossgraph_seed_123/checkpoint.pt` | 1000 | 16 | **Ablation** | Ablation study checkpoint evaluating GNN subcomponents. |
| `experiments/ablation_no_crossgraph_seed_2024/checkpoint.pt` | 1000 | 18 | **Ablation** | Ablation study checkpoint evaluating GNN subcomponents. |
| `experiments/ablation_no_crossgraph_seed_3407/checkpoint.pt` | 1000 | 20 | **Ablation** | Ablation study checkpoint evaluating GNN subcomponents. |
| `experiments/ablation_no_crossgraph_seed_42/checkpoint.pt` | 1000 | 15 | **Ablation** | Ablation study checkpoint evaluating GNN subcomponents. |
| `experiments/ablation_no_crossgraph_seed_777/checkpoint.pt` | 1000 | 18 | **Ablation** | Ablation study checkpoint evaluating GNN subcomponents. |
| `experiments/ablation_no_layernorm/checkpoint.pt` | 1000 | 17 | **Ablation** | Ablation study checkpoint evaluating GNN subcomponents. |
| `experiments/ablation_no_layernorm_seed_123/checkpoint.pt` | 1000 | 14 | **Ablation** | Ablation study checkpoint evaluating GNN subcomponents. |
| `experiments/ablation_no_layernorm_seed_2024/checkpoint.pt` | 1000 | 18 | **Ablation** | Ablation study checkpoint evaluating GNN subcomponents. |
| `experiments/ablation_no_layernorm_seed_3407/checkpoint.pt` | 1000 | 18 | **Ablation** | Ablation study checkpoint evaluating GNN subcomponents. |
| `experiments/ablation_no_layernorm_seed_42/checkpoint.pt` | 1000 | 19 | **Ablation** | Ablation study checkpoint evaluating GNN subcomponents. |
| `experiments/ablation_no_layernorm_seed_777/checkpoint.pt` | 1000 | 11 | **Ablation** | Ablation study checkpoint evaluating GNN subcomponents. |
| `experiments/ablation_no_residual/checkpoint.pt` | 1000 | 17 | **Ablation** | Ablation study checkpoint evaluating GNN subcomponents. |
| `experiments/ablation_no_residual_seed_123/checkpoint.pt` | 1000 | 19 | **Ablation** | Ablation study checkpoint evaluating GNN subcomponents. |
| `experiments/ablation_no_residual_seed_2024/checkpoint.pt` | 1000 | 8 | **Ablation** | Ablation study checkpoint evaluating GNN subcomponents. |
| `experiments/ablation_no_residual_seed_3407/checkpoint.pt` | 1000 | 17 | **Ablation** | Ablation study checkpoint evaluating GNN subcomponents. |
| `experiments/ablation_no_residual_seed_42/checkpoint.pt` | 1000 | 16 | **Ablation** | Ablation study checkpoint evaluating GNN subcomponents. |
| `experiments/ablation_no_residual_seed_777/checkpoint.pt` | 1000 | 15 | **Ablation** | Ablation study checkpoint evaluating GNN subcomponents. |
| `experiments/ablation_pooling_max/checkpoint.pt` | 1000 | 12 | **Ablation** | Ablation study checkpoint evaluating GNN subcomponents. |
| `experiments/ablation_pooling_max_seed_123/checkpoint.pt` | 1000 | 5 | **Ablation** | Ablation study checkpoint evaluating GNN subcomponents. |
| `experiments/ablation_pooling_max_seed_2024/checkpoint.pt` | 1000 | 8 | **Ablation** | Ablation study checkpoint evaluating GNN subcomponents. |
| `experiments/ablation_pooling_max_seed_3407/checkpoint.pt` | 1000 | 11 | **Ablation** | Ablation study checkpoint evaluating GNN subcomponents. |
| `experiments/ablation_pooling_max_seed_42/checkpoint.pt` | 1000 | 12 | **Ablation** | Ablation study checkpoint evaluating GNN subcomponents. |
| `experiments/ablation_pooling_max_seed_777/checkpoint.pt` | 1000 | 4 | **Ablation** | Ablation study checkpoint evaluating GNN subcomponents. |
| `experiments/run_001/checkpoint.pt` | 10 | 2 | **Smoke Test** | Smoke or validation test run on subset of 10 samples for 2 epochs. |
| `experiments/run_002/checkpoint.pt` | 1000 | 12 | **Smoke Test** | Smoke or validation test run on subset of 1000 samples for 12 epochs. |
| `experiments/run_003/checkpoint.pt` | 1000 | 9 | **Smoke Test** | Smoke or validation test run on subset of 1000 samples for 9 epochs. |
| `experiments/run_004/checkpoint.pt` | 1000 | 23 | **Smoke Test** | Smoke or validation test run on subset of 1000 samples for 23 epochs. |
| `experiments/run_005/checkpoint.pt` | 1000 | 36 | **Smoke Test** | Smoke or validation test run on subset of 1000 samples for 36 epochs. |
| `experiments/run_006/checkpoint.pt` | 1000 | 48 | **Smoke Test** | Smoke or validation test run on subset of 1000 samples for 48 epochs. |
| `experiments/run_007/checkpoint.pt` | 1000 | 12 | **Smoke Test** | Smoke or validation test run on subset of 1000 samples for 12 epochs. |
| `experiments/run_008/checkpoint.pt` | 1000 | 12 | **Smoke Test** | Smoke or validation test run on subset of 1000 samples for 12 epochs. |
| `experiments/run_009/checkpoint.pt` | 1000 | 19 | **Smoke Test** | Smoke or validation test run on subset of 1000 samples for 19 epochs. |
| `experiments/run_010/checkpoint.pt` | 1000 | 23 | **Smoke Test** | Smoke or validation test run on subset of 1000 samples for 23 epochs. |
| `experiments/run_011/checkpoint.pt` | 1000 | 6 | **Production** | Fully trained Phase 6 GNN model (N=1000 samples, 50 epochs). |
| `experiments/run_012/checkpoint.pt` | 1000 | 13 | **Production** | Fully trained Phase 6 GNN model (N=1000 samples, 50 epochs). |
| `experiments/run_013/checkpoint.pt` | 1000 | 18 | **Production** | Fully trained Phase 6 GNN model (N=1000 samples, 50 epochs). |
| `experiments/run_014/checkpoint.pt` | 1000 | 15 | **Production** | Fully trained Phase 6 GNN model (N=1000 samples, 50 epochs). |
| `experiments/run_015/checkpoint.pt` | 1000 | 38 | **Production** | Fully trained Phase 6 GNN model (N=1000 samples, 50 epochs). |
| `experiments/run_016/checkpoint.pt` | 1000 | 14 | **Production** | Fully trained Phase 6 GNN model (N=1000 samples, 50 epochs). |
| `experiments/run_017/checkpoint.pt` | 10 | 2 | **Smoke Test** | Smoke or validation test run on subset of 10 samples for 2 epochs. |
| `experiments/run_018/checkpoint.pt` | 10 | 2 | **Smoke Test** | Smoke or validation test run on subset of 10 samples for 2 epochs. |
| `experiments/run_019/checkpoint.pt` | 10 | 2 | **Smoke Test** | Smoke or validation test run on subset of 10 samples for 2 epochs. |
| `experiments/run_020/checkpoint.pt` | 10 | 2 | **Smoke Test** | Smoke or validation test run on subset of 10 samples for 2 epochs. |
| `experiments/run_021/checkpoint.pt` | 10 | 2 | **Smoke Test** | Smoke or validation test run on subset of 10 samples for 2 epochs. |
| `experiments/run_022/checkpoint.pt` | 10 | 2 | **Smoke Test** | Smoke or validation test run on subset of 10 samples for 2 epochs. |
| `experiments/run_023/checkpoint.pt` | 10 | 2 | **Smoke Test** | Smoke or validation test run on subset of 10 samples for 2 epochs. |
| `experiments/run_024/checkpoint.pt` | 10 | 2 | **Smoke Test** | Smoke or validation test run on subset of 10 samples for 2 epochs. |
| `experiments/run_025/checkpoint.pt` | 10 | 2 | **Smoke Test** | Smoke or validation test run on subset of 10 samples for 2 epochs. |
| `experiments/run_026/checkpoint.pt` | 10 | 2 | **Smoke Test** | Smoke or validation test run on subset of 10 samples for 2 epochs. |
| `experiments/run_027/checkpoint.pt` | 10 | 2 | **Smoke Test** | Smoke or validation test run on subset of 10 samples for 2 epochs. |
| `experiments/run_028/checkpoint.pt` | 10 | 2 | **Smoke Test** | Smoke or validation test run on subset of 10 samples for 2 epochs. |
| `experiments/run_029/checkpoint.pt` | 10 | 2 | **Smoke Test** | Smoke or validation test run on subset of 10 samples for 2 epochs. |
| `experiments/run_030/checkpoint.pt` | 10 | 2 | **Smoke Test** | Smoke or validation test run on subset of 10 samples for 2 epochs. |
| `experiments/run_031/checkpoint.pt` | 10 | 2 | **Smoke Test** | Smoke or validation test run on subset of 10 samples for 2 epochs. |
| `experiments/run_032/checkpoint.pt` | 10 | 2 | **Smoke Test** | Smoke or validation test run on subset of 10 samples for 2 epochs. |
| `experiments/run_033/checkpoint.pt` | 10 | 2 | **Smoke Test** | Smoke or validation test run on subset of 10 samples for 2 epochs. |
| `experiments/run_034/checkpoint.pt` | 10 | 1 | **Smoke Test** | Smoke or validation test run on subset of 10 samples for 1 epochs. |
| `experiments/run_035/checkpoint.pt` | 10 | 2 | **Smoke Test** | Smoke or validation test run on subset of 10 samples for 2 epochs. |
| `experiments/run_036/checkpoint.pt` | 10 | 2 | **Smoke Test** | Smoke or validation test run on subset of 10 samples for 2 epochs. |
| `experiments/run_037/checkpoint.pt` | 10 | 2 | **Smoke Test** | Smoke or validation test run on subset of 10 samples for 2 epochs. |
| `experiments/run_038/checkpoint.pt` | 10 | 2 | **Smoke Test** | Smoke or validation test run on subset of 10 samples for 2 epochs. |
| `experiments/run_039/checkpoint.pt` | 10 | 2 | **Smoke Test** | Smoke or validation test run on subset of 10 samples for 2 epochs. |
| `experiments/run_040/checkpoint.pt` | 10 | 2 | **Smoke Test** | Smoke or validation test run on subset of 10 samples for 2 epochs. |
| `experiments/run_041/checkpoint.pt` | 10 | 2 | **Smoke Test** | Smoke or validation test run on subset of 10 samples for 2 epochs. |
| `experiments/run_042/checkpoint.pt` | 50 | 2 | **Smoke Test** | Smoke or validation test run on subset of 50 samples for 2 epochs. |
| `experiments/run_043/checkpoint.pt` | 50 | 4 | **Smoke Test** | Smoke or validation test run on subset of 50 samples for 4 epochs. |
| `experiments/run_044/checkpoint.pt` | 50 | 5 | **Smoke Test** | Smoke or validation test run on subset of 50 samples for 5 epochs. |
| `experiments/run_045/checkpoint.pt` | 50 | 4 | **Smoke Test** | Smoke or validation test run on subset of 50 samples for 4 epochs. |
| `experiments/run_046/checkpoint.pt` | 50 | 2 | **Smoke Test** | Smoke or validation test run on subset of 50 samples for 2 epochs. |
| `experiments/run_047/checkpoint.pt` | 50 | 5 | **Smoke Test** | Smoke or validation test run on subset of 50 samples for 5 epochs. |
| `experiments/run_048/checkpoint.pt` | 50 | 4 | **Smoke Test** | Smoke or validation test run on subset of 50 samples for 4 epochs. |
| `experiments/run_049/checkpoint.pt` | 50 | 2 | **Smoke Test** | Smoke or validation test run on subset of 50 samples for 2 epochs. |
| `experiments/run_050/checkpoint.pt` | 50 | 5 | **Smoke Test** | Smoke or validation test run on subset of 50 samples for 5 epochs. |
| `experiments/run_051/checkpoint.pt` | 50 | 5 | **Smoke Test** | Smoke or validation test run on subset of 50 samples for 5 epochs. |
| `experiments/run_052/checkpoint.pt` | 50 | 4 | **Smoke Test** | Smoke or validation test run on subset of 50 samples for 4 epochs. |
| `experiments/run_053/checkpoint.pt` | 50 | 4 | **Smoke Test** | Smoke or validation test run on subset of 50 samples for 4 epochs. |
| `experiments/run_054/checkpoint.pt` | 50 | 4 | **Smoke Test** | Smoke or validation test run on subset of 50 samples for 4 epochs. |
| `experiments/run_055/checkpoint.pt` | 50 | 5 | **Smoke Test** | Smoke or validation test run on subset of 50 samples for 5 epochs. |
| `experiments/run_056/checkpoint.pt` | 50 | 2 | **Smoke Test** | Smoke or validation test run on subset of 50 samples for 2 epochs. |
| `experiments/run_057/checkpoint.pt` | 50 | 5 | **Smoke Test** | Smoke or validation test run on subset of 50 samples for 5 epochs. |
| `experiments/run_058/checkpoint.pt` | 50 | 2 | **Smoke Test** | Smoke or validation test run on subset of 50 samples for 2 epochs. |
| `experiments/run_059/checkpoint.pt` | 50 | 5 | **Smoke Test** | Smoke or validation test run on subset of 50 samples for 5 epochs. |
| `experiments/run_060/checkpoint.pt` | 50 | 2 | **Smoke Test** | Smoke or validation test run on subset of 50 samples for 2 epochs. |
| `outputs/best_protliggnn.pt` | 1000 | 13 | **Benchmark Target / Overwritten** | Main benchmark checkpoint path (restored to Production run_012). |
