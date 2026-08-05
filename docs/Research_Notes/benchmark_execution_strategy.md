# Benchmark Execution Strategy

This document details the compute, memory, and hyperparameter recommendations for executing the baseline benchmark on the target hardware.

## Performance Profiling & Estimates

Based on a measured profiling run of 100 complexes for 2 epochs on the host CPU, we observed the following performance characteristics:
- **Preprocessing Speed:** ~10-13 complexes/sec.
- **Training Throughput:** ~1.6 seconds per epoch for 100 complexes.

Using these empirical values, we project runtimes for different dataset scales below:

| Metric / Parameter | 100 Samples (Measured) | 1,000 Samples (Est.) | 5,000 Samples (Est.) |
|---|---|---|---|
| **Graph Preprocessing Time** | ~9 seconds | ~1.6 minutes | ~8.3 minutes |
| **Training Time (per epoch)** | ~1.6 seconds | ~16 seconds | ~80 seconds |
| **Single Run (40 epochs)** | ~1.2 minutes | ~12.2 minutes | ~61.6 minutes |
| **5-Seed Baseline (no-crossgraph)** | ~6 minutes | ~61 minutes | ~308 minutes (~5.1 hours) |
| **5-Seed Model (crossgraph)** | ~6 minutes | ~61 minutes | ~308 minutes (~5.1 hours) |
| **Total Benchmark Suite** | ~12 minutes | **~2 hours** | **~10.2 hours** |

---

## Resource & Hyperparameter Recommendations

### 1. Device
- **Recommendation:** `--device cpu`
- **Rationale:** No CUDA-enabled PyTorch package is available for Python 3.14 on Windows, rendering GPU execution impossible.

### 2. Batch Size
- **Recommendation:** `--batch_size 8` (Keep unchanged)
- **Rationale:** Keeps memory consumption low and maintains scientific compatibility with historical runs. Larger batch sizes do not speed up CPU training significantly.

### 3. DataLoader Workers (`--num_workers`)
- **Recommendation:** `--num_workers 0`
- **Rationale:** Windows uses the `spawn` start method for multiprocessing, which copies the entire Python process. Since available RAM is constrained (~3.88 GB free), spawning multiple data loader processes will duplicate the massive in-memory graph dataset, causing memory thrashing, page swapping, and potential Out-Of-Memory (OOM) crashes. Setting workers to `0` ensures data loading is single-threaded in the main process, conserving memory.

### 4. Memory Pinning (`--pin_memory`)
- **Recommendation:** Disable (Do not use `--pin_memory`)
- **Rationale:** Memory pinning is used to accelerate CPU-to-GPU memory copies. For CPU-only training, it offers no benefit and increases RAM overhead.

### 5. Mixed Precision (AMP)
- **Recommendation:** Disable
- **Rationale:** Automatic Mixed Precision (AMP) is designed for CUDA GPUs. PyTorch CPU mixed precision (BFloat16) is only beneficial on modern server CPUs with AVX-512 / AMX extensions. On standard client CPUs, it can introduce numerical instability and slowdowns.

### 6. Checkpoint and Resume Frequency
- **Checkpointing:** Save checkpoint at the end of every epoch where validation RMSE improves (this is default behavior in `protliggnn_train.py`).
- **Resumability:** Set `--resume latest` in case of interruptions to allow resuming training from the last saved epoch of the active run.
