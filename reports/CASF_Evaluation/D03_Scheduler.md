# D03 Scheduler Study: CASF-2016

## Learning Rate Scheduler Sweep
| Scheduler | Optimizer | Learning Rate | Val RMSE | CASF Pearson (local test) | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **None** (trial 012) | Adam | 1e-3 | 1.5902 | 0.4117 | **Measured** |
| **Cosine Annealing** (trial 000) | AdamW | 1.89e-4 | **1.5153** | **0.5039** | **Measured (Best)** |
| **ReduceLROnPlateau** (trial 007) | RAdam | 2.22e-4 | 1.5333 | 0.4876 | **Measured** |
| **OneCycle** | - | - | - | - | *Evidence not available* |

## Verdict
- Cosine Annealing learning rate scheduling paired with AdamW optimizer is superior to constant learning rate, achieving a local validation RMSE of **1.5153** and a Pearson correlation of **0.5039** (compared to 1.5902 and 0.4117 for baseline scheduler-free Adam).
