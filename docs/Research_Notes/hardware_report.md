# Hardware Discovery Report

This report documents the hardware and environment discovery performed on the local execution host.

## System Profile

- **Operating System:** Windows 11 (Version 10.0.26200)
- **Architecture:** AMD64 (64-bit Windows)
- **Python Version:** 3.14.3
- **PyTorch Version:** 2.13.0+cpu

## Compute Resources

### CPU (Central Processing Unit)
- **Processor:** Intel64 Family 6 Model 154 Stepping 3 (Intel Core CPU)
- **Physical Cores:** 8
- **Logical Cores (Threads):** 12

### Memory (RAM)
- **Total Physical Memory:** 15.73 GB
- **Available Memory:** ~3.88 GB (Note: Host has high memory utilization; free RAM is limited to ~4 GB)

### GPU (Graphics Processing Unit)
- **Discrete GPU:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (Driver: 560.94, CUDA Version: 12.6)
- **Integrated GPU:** Intel(R) UHD Graphics

---

## Acceleration Backend & Compatibility Analysis

| Backend | Detected Hardware | PyTorch Support | Status |
|---|---|---|---|
| **CUDA** | NVIDIA RTX 3050 | No | **Unsupported** in Python 3.14 on Windows (no precompiled `cp314` wheels on PyTorch index) |
| **ROCm** | None | No | **Unsupported** (AMD GPU required) |
| **Intel XPU** | Intel UHD | No | **Unsupported** (requires Intel Arc or dedicated Xe GPU and `intel-extension-for-pytorch`) |
| **DirectML** | NVIDIA / Intel | No | **Unsupported** (no `torch-directml` module available for Python 3.14 yet) |
| **CPU (Fallback)** | Intel Core | Yes | **Supported** (utilizes CPU backend `PyTorch 2.13.0+cpu`) |

### Package Limitation Analysis
Although the machine features a physical NVIDIA GeForce RTX 3050 GPU with CUDA capability (Driver supports up to CUDA 12.6), the Python interpreter version is **Python 3.14.3**. As a result:
1. PyTorch has not yet released CUDA-enabled binary wheels (`+cu*` tags) for Python 3.14 on Windows.
2. The virtual environment falls back to `2.13.0+cpu`, which executes exclusively on the CPU.
3. PyTorch Geometric (`torch-geometric`) and all graph computations are executed via CPU tensors.

---

## Machine Classification

**Classification:** `CPU ONLY` (for PyTorch execution)

Due to the python package ecosystem constraint on Python 3.14, the hardware can only be utilized for **CPU-only** PyTorch execution.

## Optimal Execution Strategy Recommendation

1. **Parallel Workers:** Set `--num_workers 0` or a very low value (e.g., `2` at most) because graph construction in Python is CPU-bound, and Windows multiprocessing suffers from spawn overhead. Additionally, free RAM is low (~4 GB), so spawning multiple data loader workers will trigger out-of-memory (OOM) swapping.
2. **Batch Size:** A batch size of `8` is safe. Larger batches will increase memory usage without improving throughput on CPU.
3. **Execution Device:** Force `--device cpu` explicitly.
4. **Data Size Cap:** Because training is restricted to CPU, a full 5,000-sample, 40-epoch benchmark run across 5 seeds will take too long (estimated at ~15-20 hours). We recommend producing a **Validated Baseline v0.9** with a subset of the dataset (`--max_samples 1000`) and a lower epoch count (`--epochs 15`), which will allow validation within a reasonable timeframe (~2 hours) while maintaining relative comparative integrity.
