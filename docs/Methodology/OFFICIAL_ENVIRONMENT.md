# ProtLigGNN Official Research Environment v1.0

This document defines the official software stack, system requirements, installation guide, and verification steps for the ProtLigGNN research platform. Every official benchmark, experiment, and model comparison must execute within this environment to guarantee numerical consistency and reproducibility.

---

## 1. System Requirements & Software Stack

### Supported Operating Systems
- **Windows:** Windows 10 / 11 (64-bit)
- **Linux:** Ubuntu 20.04 / 22.04 LTS (64-bit)

### Compute Hardware Requirements
- **CPU:** Intel or AMD x86_64 CPU (6+ physical cores recommended)
- **System Memory:** 16 GB RAM minimum
- **Discrete GPU:** NVIDIA GPU (Ampere/Ada Lovelace laptop or desktop GPU recommended)
- **VRAM:** 6 GB minimum (8 GB+ recommended)
- **NVIDIA GPU Driver:** Version 550.x or higher (supporting CUDA compute capability >= 8.6)

### Target Version Pinning

| Dependency | Official Version | Installation Source / Index |
|---|---|---|
| **Python** | `3.11.9` | Official Installer / WinGet |
| **PyTorch** | `2.3.1+cu121` | https://download.pytorch.org/whl/cu121 |
| **CUDA Toolkit** | `12.1` | Embedded inside PyTorch package |
| **PyTorch Geometric (PyG)** | `2.8.0` | PyPI |
| **torch-scatter** | `2.1.2+pt23cu121` | https://data.pyg.org/whl/torch-2.3.1+cu121.html |
| **torch-sparse** | `0.6.18+pt23cu121` | https://data.pyg.org/whl/torch-2.3.1+cu121.html |
| **torch-cluster** | `1.6.3+pt23cu121` | https://data.pyg.org/whl/torch-2.3.1+cu121.html |
| **torch-spline-conv** | `1.2.2+pt23cu121` | https://data.pyg.org/whl/torch-2.3.1+cu121.html |
| **NumPy** | `1.26.4` (Pinned) | PyPI |
| **SciPy** | `1.17.1` | PyPI |
| **pandas** | `3.0.3` | PyPI |
| **scikit-learn** | `1.9.0` | PyPI |
| **Matplotlib** | `3.11.0` | PyPI |
| **RDKit** | `2026.03.3` | PyPI |
| **BioPython** | `1.87` | PyPI |

> [!IMPORTANT]
> **Why NumPy is Pinned to 1.26.4:**  
> NumPy 2.x introduces major API and binary ABI changes that are incompatible with PyTorch 2.3.1 and the compiled PyG extensions. Installing NumPy 2.x will trigger segmentation faults and runtime crashes in graph operations. NumPy must remain pinned to `<2.0` (specifically `1.26.4`).

---

## 2. Installation Guide (Windows PowerShell)

### Step 1: Install Python 3.11.9
Using Windows Package Manager (no administrative rights required):
```powershell
winget install Python.Python.3.11 --scope user --accept-source-agreements --accept-package-agreements
```

### Step 2: Create a Fresh Virtual Environment
Navigate to the repository root directory and create the `venv`:
```powershell
C:\Users\Harshhhh\AppData\Local\Programs\Python\Python311\python.exe -m venv venv
```

### Step 3: Upgrade Basic Tools
```powershell
.\venv\Scripts\python -m pip install --upgrade pip setuptools wheel
```

### Step 4: Install CUDA-Enabled PyTorch
```powershell
.\venv\Scripts\pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

### Step 5: Pin NumPy 1.x
```powershell
.\venv\Scripts\pip install numpy==1.26.4
```

### Step 6: Install PyTorch Geometric & Precompiled Extensions
```powershell
.\venv\Scripts\pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
```

### Step 7: Install Biology & Supporting Dependencies
```powershell
.\venv\Scripts\pip install rdkit biopython scikit-learn matplotlib pyyaml pandas pytest tqdm
```

---

## 3. Environment Verification

To verify that all dependencies are compiled, linked, and executing correctly on the GPU, run the verification script:
```powershell
.\venv\Scripts\python verify_environment.py
```

Expected output:
```text
==================================================
      ProtLigGNN Environment Verification         
==================================================
[+] Python Version: 3.11.9
  -> PASS: Python version is 3.11
[+] PyTorch Version: 2.3.1+cu121
  -> CUDA Available: True
  -> GPU Name: NVIDIA GeForce RTX 3050 6GB Laptop GPU
  -> CUDA Version: 12.1
  -> CUDA Tensor computation successful: cuda:0
  -> GPU Memory allocated: 8.13 MB, reserved: 22.00 MB
  -> AMP/Autocast context validation: PASS
[+] PyTorch Geometric Version: 2.8.0
  -> torch-scatter: PASS
  -> torch-sparse: PASS
  -> torch-cluster: PASS
  -> torch-spline-conv: PASS
  -> PyG Extensions load: PASS
[+] RDKit: PASS (mol created with conformer: 3 atoms)
[+] BioPython PDBParser: PASS
[+] NumPy: Version 1.26.4 (PASS)
[+] SciPy: Version 1.17.1 (PASS)
[+] pandas: Version 3.0.3 (PASS)
[+] scikit-learn: Version 1.9.0 (PASS)
[+] matplotlib: Version 3.11.0 (PASS)
==================================================
RESULT: PASS
```

---

## 4. Troubleshooting & Known Limitations

### CUDA Not Available (`CUDA Available: False`)
1. Ensure your NVIDIA GPU driver is updated (minimum 550.x on Windows). Run `nvidia-smi` to verify.
2. Confirm you installed PyTorch from the `--index-url https://download.pytorch.org/whl/cu121` url. If you installed via PyPI defaults, you will receive the CPU-only version. Run `pip uninstall torch` and reinstall with the correct index URL.

### Segmentation Faults on Graph Operations
This is caused by a NumPy 2.x ABI mismatch. Force downgrade NumPy to 1.26.4:
```powershell
.\venv\Scripts\pip install numpy==1.26.4 --force-reinstall
```

### Windows Multi-Processing memory overhead
Windows uses the `spawn` method, duplicating RAM. If your host has less than 8 GB of free RAM, set `--num_workers 0` in all training commands to prevent virtual memory swapping and sluggish performance.

### Reproducibility warnings on CUDA >= 10.2
To force cuBLAS to execute GEMM and attention layers fully deterministically, ensure the environment variable `CUBLAS_WORKSPACE_CONFIG` is set:
```powershell
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"
```
*(This is set automatically inside the training codebase).*
