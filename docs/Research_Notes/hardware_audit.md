# Hardware Audit Report

This report documents the hardware capabilities, driver versions, and compute resource limits of the host machine for the ProtLigGNN research project.

## Hardware Summary

### CPU (Central Processing Unit)
- **Model:** Intel Core CPU (Intel64 Family 6 Model 154 Stepping 3)
- **Physical Cores:** 8
- **Logical Cores (Threads):** 12

### Memory (RAM)
- **Total Physical RAM:** 15.73 GB
- **Available RAM:** ~3.88 GB (Note: High host memory utilization; free memory is limited to ~4 GB)

### GPU (Graphics Processing Unit)
- **Discrete GPU:** NVIDIA GeForce RTX 3050 Laptop GPU
- **VRAM:** 6 GB GDDR6
- **GPU Driver:** 560.94 (WDDM Model)
- **CUDA Driver Support:** Up to CUDA Version 12.6

---

## Software Environment Status

| Library/Component | Current Environment (Before Migration) | Target Environment (After Migration) | Status |
|---|---|---|---|
| **Python** | 3.14.3 | 3.11.9 | **Downgrading** to stable, compatible version |
| **PyTorch** | 2.13.0+cpu | 2.3.1+cu121 | **Upgrading** to CUDA-enabled GPU version |
| **CUDA Toolkit** | None (CPU Fallback) | 12.1 (Embedded via PyTorch) | **Enabled** GPU acceleration |
| **PyTorch Geometric** | 2.6.1 (CPU execution) | 2.5.3 (CUDA binary extensions) | **Enabled** GPU graph operations |
| **RDKit** | 2024.3.6 | 2024.3.6 (or latest) | Stable |
| **BioPython** | 1.84 | 1.84 (or latest) | Stable |
| **NumPy** | 2.2.3 | 1.26.4 (or latest compatible v1) | Stable |
| **SciPy** | 1.15.1 | 1.12.0 (or latest compatible) | Stable |
| **scikit-learn** | 1.6.1 | 1.4.2 (or latest compatible) | Stable |
| **pandas** | 2.2.3 | 2.2.2 (or latest compatible) | Stable |
| **Matplotlib** | 3.10.0 | 3.8.4 (or latest compatible) | Stable |

---

## Compute Acceleration Capabilities

### Supported Acceleration
- **CUDA 12.1 / 12.4 (NVIDIA GPU):** The NVIDIA GeForce RTX 3050 has full compute capability (8.6) to run CUDA-accelerated tensors, graph neural network message passing, and autodiff forward/backward passes.

### Unsupported Acceleration
- **ROCm:** Not supported (no AMD GPU present).
- **Intel XPU / oneAPI:** Not supported for deep learning acceleration (integrated Intel UHD is too low-spec).

---

## Resource Limitations & Recommendations

1. **VRAM Capacity (6 GB):** The RTX 3050 is a laptop GPU with 6 GB VRAM. 
   - *Recommendation:* Keep training batch size at `8` or `16`. If larger effective batch sizes are required in future phases, implement **gradient accumulation** rather than increasing the physical batch size to avoid GPU Out-Of-Memory (OOM) errors.
2. **System RAM (15.73 GB total, ~3.88 GB free):** Low free physical memory is a critical bottleneck.
   - *Recommendation:* Set `--num_workers 0` or at most `2` when running training. Using high worker counts on Windows duplicates memory per process and will crash the host.
3. **Multiprocessing on Windows:** Windows lacks copy-on-write `fork` semantics.
   - *Recommendation:* Always utilize clean spawn-safe multiprocessing guards (`if __name__ == "__main__":`).
