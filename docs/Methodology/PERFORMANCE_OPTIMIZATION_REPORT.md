# ProtLigGNN Performance Optimization Report

This report analyzes training bottlenecks and proposes optimizations that preserve scientific correctness and benchmark protocol alignment.

---

## 1. GPU Wait Time & Starvation Analysis

During active training:
- **GPU Wait Time:** `0.0372 seconds` (`1.2%` of training epoch).
- **Assessment:** The GPU is **not** starved of data during the training loop. Because all pocket and ligand graphs are fully pre-constructed and cached in system RAM (`dataset.samples`) prior to dataloader instantiation, batch iteration is extremely fast (`0.0372s` total per epoch) and introduces zero file I/O overhead.
- **Worker Configuration:** Spawning multiple processes (`--num_workers > 0`) is **not** required for dataloading performance on this dataset size, avoiding Windows multiprocessing overhead and potential memory swapping.

---

## 2. Primary Bottleneck Identification

### Bottleneck A: Startup Graph Construction (One-time Overhead)
- **Time:** `25.75s` (200 samples) | Scaled: `128.8s` (1,000 samples), `644.0s` (5,000 samples).
- **Cause:** Residue pocket extraction and RDKit molecule conversion are executed in a single-threaded Python loop at dataset initialization. This blocks training startup.
- **Classification:** **Graph construction / Python overhead**.

### Bottleneck B: Training GPU Underutilization (Active Loop)
- **Average GPU Util:** `10.0%` (Peak: `13.0%`).
- **Cause:** The ProtLigGNN model (GAT/GCN) is very small, and the batch size of 8 is small. This causes PyTorch to execute small CUDA kernels, where kernel execution time is shorter than Python overhead and kernel launch latency.
- **Classification:** **Python overhead / CUDA kernel launch latency**.

---

## 3. Recommended Optimizations (Correctness Preserving)

The following optimizations can be implemented to accelerate training without modifying the benchmark dataset, splits, seeds, or metrics:

### Optimization 1: Multi-processed Preprocessing at Startup
- **Approach:** Use Python `multiprocessing.Pool` or `concurrent.futures.ProcessPoolExecutor` in `discover_complex_records` and `PDBbindPairDataset` to build ligand and residue pocket graphs in parallel across CPU cores.
- **Impact:** Reduces startup graph construction time from `10.7 minutes` to `~1.5 minutes` on an 8-core CPU.

### Optimization 2: Pre-save Graph Dataset to Disk
- **Approach:** Serialize the preprocessed list of PyG graph objects (e.g. `processed_dataset.pt`) using `torch.save` once. Subsequent benchmark runs can load this serialized file directly.
- **Impact:** Eliminates the startup graph construction phase entirely, reducing startup time from `10.7 minutes` to `~1.2 seconds`.

### Optimization 3: Leverage Model Compilation (`torch.compile`)
- **Approach:** Apply `torch.compile(model)` in PyTorch 2.x to fuse GNN operations and GAT message passing steps.
- **Impact:** Fuses CUDA kernels, reducing Python execution overhead and kernel launch bottlenecks, resulting in a ~1.5x-2.0x active speedup.

### Optimization 4: GPU Batch Size Scale with Gradient Accumulation
- **Approach:** If we increase batch size to `32` to saturate the GPU, we can use gradient accumulation (e.g., accumulation steps = 4) to preserve the effective batch size of 8 if we want to match exact gradient step updates, or promote the benchmark to v1.1.
- **Impact:** Increases GPU utilization from 10% to ~40%, reducing active epoch time.
