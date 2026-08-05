import time
import sys
import torch
from pathlib import Path
from torch_geometric.data import Data, Batch
import psutil

# Add import paths
sys.path.append(str(Path.cwd()))

from protliggnn_train import (
    discover_complex_records,
    PDBbindPairDataset,
    ProtLigGNN,
    make_loader,
    run_epoch,
    compute_metrics,
    LIGAND_FEATURE_DIM,
    PROTEIN_FEATURE_DIM
)

def get_gpu_usage():
    try:
        import subprocess
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        parts = res.stdout.strip().split(",")
        gpu_util = float(parts[0].strip())
        gpu_mem_used = float(parts[1].strip())
        return gpu_util, gpu_mem_used
    except Exception:
        return 0.0, 0.0

def main():
    print("==================================================")
    print("         ProtLigGNN Training Pipeline Profiler     ")
    print("==================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Dataset Loading & Graph Construction Profiling
    print("[*] Profiling Dataset Loading and Graph Construction...")
    t_start = time.perf_counter()
    
    # We will load 200 samples for profiling to get stable metrics quickly
    data_dir = Path("data/pdbbind2020")
    
    # Time index parsing & discovery
    t_idx_start = time.perf_counter()
    records = discover_complex_records(data_dir, max_samples=200)
    t_idx_end = time.perf_counter()
    idx_duration = t_idx_end - t_idx_start
    print(f"  -> Index parsing duration: {idx_duration:.4f} seconds")
    
    # Time graph construction on first 200 samples
    t_graph_start = time.perf_counter()
    dataset = PDBbindPairDataset(records)
    t_graph_end = time.perf_counter()
    total_data_load = t_graph_end - t_start
    print(f"  -> Total dataset prep + graph construction duration: {total_data_load:.4f} seconds ({total_data_load/200:.4f}s/complex)")

    # 2. Split and Dataloader
    print("[*] Creating DataLoaders...")
    indices = list(range(len(dataset)))
    train_idx = indices[:160]
    val_idx = indices[160:180]
    
    t_loader_start = time.perf_counter()
    train_loader = make_loader(dataset.samples, train_idx, batch_size=8, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = make_loader(dataset.samples, val_idx, batch_size=8, shuffle=False, num_workers=0, pin_memory=False)
    t_loader_end = time.perf_counter()
    print(f"  -> DataLoader creation: {t_loader_end - t_loader_start:.4f} seconds")

    # 3. Model instantiation
    model = ProtLigGNN(
        ligand_dim=LIGAND_FEATURE_DIM,
        protein_dim=PROTEIN_FEATURE_DIM,
        no_crossgraph=False
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    # 4. Training Loop Profiling
    print("[*] Profiling Epoch Stages (Train)...")
    
    stages = {
        "dataloader": 0.0,
        "transfer": 0.0,
        "forward": 0.0,
        "loss": 0.0,
        "backward": 0.0,
        "optimizer": 0.0
    }
    
    gpu_utils = []
    gpu_mems = []
    cpu_utils = []
    
    model.train()
    
    # Run 1 epoch and measure stage timings
    epoch_start = time.perf_counter()
    
    t_iter = time.perf_counter()
    for batch_idx, (ligand_batch, protein_batch, labels, batch_pdb_ids) in enumerate(train_loader):
        stages["dataloader"] += time.perf_counter() - t_iter
        
        # GPU Query
        gpu_u, gpu_m = get_gpu_usage()
        gpu_utils.append(gpu_u)
        gpu_mems.append(gpu_m)
        cpu_utils.append(psutil.cpu_percent())
        
        # CPU to GPU Transfer
        t_trans = time.perf_counter()
        ligand_batch = ligand_batch.to(device)
        protein_batch = protein_batch.to(device)
        labels = labels.to(device)
        stages["transfer"] += time.perf_counter() - t_trans
        
        # Forward pass
        t_fwd = time.perf_counter()
        preds = model(ligand_batch, protein_batch)
        stages["forward"] += time.perf_counter() - t_fwd
        
        # Loss computation
        t_loss = time.perf_counter()
        loss = criterion(preds, labels)
        stages["loss"] += time.perf_counter() - t_loss
        
        # Backward pass
        t_bwd = time.perf_counter()
        optimizer.zero_grad()
        loss.backward()
        stages["backward"] += time.perf_counter() - t_bwd
        
        # Optimizer step
        t_opt = time.perf_counter()
        optimizer.step()
        stages["optimizer"] += time.perf_counter() - t_opt
        
        t_iter = time.perf_counter()
        
    epoch_duration = time.perf_counter() - epoch_start
    print(f"  -> Active training epoch duration: {epoch_duration:.4f} seconds")

    # 5. Validation Loop Profiling
    print("[*] Profiling Validation Epoch...")
    t_val_start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for ligand_batch, protein_batch, labels, batch_pdb_ids in val_loader:
            ligand_batch = ligand_batch.to(device)
            protein_batch = protein_batch.to(device)
            preds = model(ligand_batch, protein_batch)
    val_duration = time.perf_counter() - t_val_start
    print(f"  -> Validation epoch duration: {val_duration:.4f} seconds")

    # 6. Checkpoint Serialization Profiling
    print("[*] Profiling Checkpoint Serialization...")
    t_ckpt_start = time.perf_counter()
    checkpoint_data = {
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "stats": {"val_rmse": 1.25}
    }
    torch.save(checkpoint_data, "experiments/run_profiler_temp.pt")
    ckpt_duration = time.perf_counter() - t_ckpt_start
    print(f"  -> Checkpoint write duration: {ckpt_duration:.4f} seconds")
    
    # Clean up temp checkpoint
    if Path("experiments/run_profiler_temp.pt").exists():
        Path("experiments/run_profiler_temp.pt").unlink()

    # Reporting Metrics
    print("\n================== PROFILE RESULTS ==================")
    total_active_epoch = sum(stages.values())
    print(f"Total training active stages time: {total_active_epoch:.4f}s")
    for stage, t_val in stages.items():
        pct = (t_val / total_active_epoch) * 100
        print(f"  Stage: {stage:<12} | Time: {t_val:.4f}s ({pct:.1f}%)")
    
    print(f"\nSystem Telemetry:")
    print(f"  Avg CPU Util   : {sum(cpu_utils)/len(cpu_utils):.1f}%")
    print(f"  Peak CPU Util  : {max(cpu_utils):.1f}%")
    if gpu_utils:
        print(f"  Avg GPU Util   : {sum(gpu_utils)/len(gpu_utils):.1f}%")
        print(f"  Peak GPU Util  : {max(gpu_utils):.1f}%")
        print(f"  Avg GPU VRAM   : {sum(gpu_mems)/len(gpu_mems):.1f} MB")
    
    # Calculate GPU idle/wait time
    # The time GPU is waiting for CPU is dataloader stage time + dataset prep time!
    gpu_wait_time = stages["dataloader"]
    gpu_idle_pct = (gpu_wait_time / epoch_duration) * 100
    print(f"  GPU Wait Time  : {gpu_wait_time:.4f}s ({gpu_idle_pct:.1f}% of training epoch duration)")
    
    # Throughput
    throughput = len(train_idx) / epoch_duration
    print(f"  Throughput     : {throughput:.2f} samples/sec")
    print(f"  Batch rate     : {len(train_loader) / epoch_duration:.2f} batches/sec")
    print("==================================================")

if __name__ == "__main__":
    main()
