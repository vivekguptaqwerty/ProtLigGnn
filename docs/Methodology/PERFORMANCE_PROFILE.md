# ProtLigGNN Training Pipeline Performance Profile

This document details the performance profiling results of the ProtLigGNN training pipeline on the standardized GPU environment. Measurements were obtained by running a 1-epoch profiling run over 200 complexes on an **NVIDIA GeForce RTX 3050 Laptop GPU** (6 GB VRAM) and an **8-core Intel CPU**.

---

## 1. Time Breakdown per Epoch (Training Stage)

The total active execution time for a training epoch (160 samples, batch size 8, 20 steps) is **3.0555 seconds** (active training stage computation: **1.9624 seconds**).

| Pipeline Stage | Active Time (s) | Proportion (%) | Description |
|---|---|---|---|
| **DataLoader Iteration** | `0.0372s` | `1.9%` | Fetching pre-constructed batches from RAM |
| **CPU → GPU Transfer** | `0.0136s` | `0.7%` | Moving ligand/protein pocket graphs to VRAM |
| **Forward Pass** | `1.0544s` | `53.7%` | GAT, GCN, cross-attention, and MLP regression layers |
| **Loss Computation** | `0.0202s` | `1.0%` | MSE loss evaluation |
| **Backward Pass** | `0.7411s` | `37.8%` | Autograd gradient calculation |
| **Optimizer Step** | `0.0959s` | `4.9%` | Parameter weight updates via Adam |

---

## 2. Overall Pipeline Phase Durations

- **Dataset Index Discovery:** `0.011s`
- **Graph Construction (200 complexes):** `25.7548s` (`0.1288s` per complex)
- **DataLoader Creation:** `0.0003s`
- **Active Training Epoch (20 batches):** `3.0555s`
- **Validation Epoch (20 samples):** `0.0435s`
- **Checkpoint Writing:** `0.0332s`

---

## 3. System Telemetry & Resource Utilization

### GPU Utilization & Memory
- **Average GPU Utilization:** `10.0%`
- **Peak GPU Utilization:** `13.0%`
- **GPU VRAM Allocation:** `205.4 MB` (Very small footprint)
- **GPU Wait Time:** `0.0372s` (`1.2%` of training epoch)
- **GPU Idle Percentage:** `98.8%` (Not compute bound, heavily constrained by kernel execution overhead)

### CPU Utilization & Threads
- **Average CPU Utilization:** `60.0%`
- **Peak CPU Utilization:** `78.2%`
- **Active Threads:** `12` (logical processors utilized during graph construction/multiprocessing)
- **Time Preparing Batches:** `0.0372s` (dataloader latency is negligible due to RAM caching)

---

## 4. Throughput & Execution Stats

- **Throughput:** `52.37 samples/second`
- **Batch Rate:** `6.55 batches/second`
- **Active Step Time:** `0.098s per batch`
- **One-time Startup Overhead:** `128.8s` (for 1,000 samples) / `644.0s` (for 5,000 samples)

---

## 5. Primary Bottleneck Classification

- **Active Training Bottleneck:** **GPU compute/Python overhead** (launching many tiny PyG message passing and attention kernels on a small batch size of 8).
- **Startup Bottleneck:** **Graph construction** (single-threaded PDB residue pocket extraction and RDKit molecule processing at initialization).
