# Benchmark Status Report

This report documents the current execution status and state of the ProtLigGNN benchmark.

---

## 1. Active Execution Diagnostics

1. **Is `benchmark.py` currently executing?**
   **YES.** It is actively running in the background as task `task-711` (command: `.\venv\Scripts\python benchmark.py --no_crossgraph`).
2. **If not, why did it stop?**
   N/A (it is running).
3. **Was the benchmark ever started after the last environment migration?**
   **YES.** It was launched in this session and is currently executing its first stage (No-Crossgraph, Seed 42).
4. **If it exited, what was the exit code?**
   N/A (active).
5. **Was an exception raised?**
   **NO.** There are no tracebacks or runtime errors in the logs.
6. **Which benchmark artifacts already exist?**
   - The benchmark folder: `experiments/benchmark_2026-07-11T175719_0000/`
   - Active configuration plan: `experiments/benchmark_2026-07-11T175719_0000/benchmark_plan.json`
   - Progress tracker: `experiments/benchmark_progress.json`
   - Real-time logging: `benchmark_live.log`
7. **Which experiment directories have completed successfully?**
   - In the current benchmark: None (Seed 42 is active).
   - In previous validation test runs: `experiments/run_030`, `experiments/run_031`, `experiments/run_032`, `experiments/run_033`, and `experiments/run_034` successfully completed.
8. **What is the status of `experiments/registry.csv`?**
   Active and valid. It contains registry logs of all past successful validation and smoke runs, and will register the benchmark seeds upon their respective completions.
9. **Is the repository currently waiting for user approval?**
   **NO.** It is actively computing.
10. **Can the benchmark safely resume from existing checkpoints?**
    **YES.** If interrupted, running `benchmark.py` will detect `experiments/benchmark_progress.json` automatically, reuse the existing run directories, pass the `--resume` flag to PyTorch to continue from the last saved `checkpoint.pt` for the active seed, and skip already completed seeds.

---

## 2. Current Progress Snapshot

- **Current Active Seed:** `42`
- **Active Run Directory:** `experiments/run_035`
- **Phase:** Cache Building & Molecular Graph Processing (constructing the 1,000-complex pocket graphs). The CPU is currently running RDKit atom-feature loops.
