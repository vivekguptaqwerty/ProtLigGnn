# A12 Regression Test Report: CASF-2016

## Execution Summary
- **Test Command**: `.env\Scripts\python -m pytest`
- **Total Collected Tests**: 103
- **Passed Tests**: 103
- **Failed Tests**: 0
- **Warnings**: 3 (Deprecation warnings for PyG/Torch)
- **Execution Time**: 22.04 seconds

## Verification Log
All 103 tests in the test suite pass successfully:
- `tests/test_attention_bias.py`: PASS (3/3)
- `tests/test_baseline_equivalence.py`: PASS (2/2)
- `tests/test_benchmark_equivalence.py`: PASS (4/4)
- `tests/test_checkpoint.py`: PASS (2/2)
- `tests/test_soft_routing.py`: PASS (6/6)
- `tests/test_uncertainty.py`: PASS (7/7)
- `tests/test_robustness.py`: PASS (6/6)
- `tests/test_geometry_interaction.py`: PASS (9/9)
- `tests/test_equivariant.py`: PASS (9/9)
- `tests/test_foundation_hybrid.py`: PASS (5/5)
- `tests/test_foundation_ligand.py`: PASS (6/6)
- `tests/test_foundation_protein.py`: PASS (5/5)

No regression or module import errors exist in the codebase.
