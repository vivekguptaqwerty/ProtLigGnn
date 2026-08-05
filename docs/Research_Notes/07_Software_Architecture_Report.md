# 07. Software Architecture Report
**Experiment ID**: P5.5-INTEGRATION  

## Architecture Overview
The integrated repository structure uses clean, non-intrusive wrapping:
- `core/` stores shared indices and constants.
- `inference/` isolates the prediction pipelines.
- `configs/` consolidates hyperparameters.
- `utils/` coordinates file and log utilities.
