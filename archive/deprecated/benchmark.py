import argparse
import subprocess
import sys
import os
import time
import re
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional

import psutil
import yaml

from benchmarking import DEFAULT_SEEDS, read_json, resolve_experiment_dir, utc_now, write_json, write_results_summary


def parse_seed_argument(values: List[str]) -> List[int]:
    if not values:
        return DEFAULT_SEEDS
    if len(values) == 1:
        value = values[0]
        if "," in value:
            return [int(part.strip()) for part in value.split(",") if part.strip()]
        count = int(value)
        if count <= len(DEFAULT_SEEDS):
            return DEFAULT_SEEDS[:count]
        seeds = list(DEFAULT_SEEDS)
        next_seed = 9001
        while len(seeds) < count:
            seeds.append(next_seed)
            next_seed += 7919
        return seeds
    return [int(value) for value in values]


def parse_args():
    # 1. Parse --config pre-argument
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="benchmark_config.yaml", help="Path to YAML config file.")
    pre_args, remaining_args = pre_parser.parse_known_args()

    # 2. Load YAML defaults if available
    defaults = {}
    config_path = Path(pre_args.config)
    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    defaults = loaded
        except Exception as e:
            print(f"Warning: Failed to load config file: {e}")

    # 3. Build main parser populated with YAML defaults
    parser = argparse.ArgumentParser(description="Run a multi-seed ProtLigGNN benchmark.")
    parser.add_argument("--config", type=str, default=pre_args.config, help="Path to YAML config file.")
    parser.add_argument("--data_dir", type=str, default=defaults.get("data_dir", "data/pdbbind2020"))
    parser.add_argument("--device", type=str, default=defaults.get("device", "cpu"))
    parser.add_argument("--epochs", type=int, default=defaults.get("epochs", 50))
    parser.add_argument("--batch_size", type=int, default=defaults.get("batch_size", 8))
    parser.add_argument("--lr", type=float, default=defaults.get("lr", 1e-3))
    parser.add_argument("--weight_decay", type=float, default=defaults.get("weight_decay", 1e-5))
    parser.add_argument("--patience", type=int, default=defaults.get("patience", 10))
    parser.add_argument("--max_samples", type=int, default=defaults.get("max_samples", None))
    parser.add_argument("--run_name", type=str, default=defaults.get("run_name", "protliggnn_benchmark"))
    parser.add_argument("--no_crossgraph", action="store_true", default=defaults.get("no_crossgraph", False))
    parser.add_argument("--save_path", type=str, default=None)
    
    # Parse default seeds list
    default_seeds_arg = defaults.get("seeds")
    if default_seeds_arg is not None:
        if isinstance(default_seeds_arg, list):
            default_seeds_arg = [str(s) for s in default_seeds_arg]
        else:
            default_seeds_arg = [str(default_seeds_arg)]
    else:
        default_seeds_arg = [str(len(DEFAULT_SEEDS))]

    parser.add_argument("--seeds", nargs="*", default=default_seeds_arg)
    parser.add_argument(
        "--split_strategy",
        choices=["random", "fixed", "scaffold", "protein_family", "temporal"],
        default=defaults.get("split_strategy", "random"),
    )
    parser.add_argument("--split_dir", type=str, default=defaults.get("split_dir", "splits"))
    parser.add_argument("--fixed_split_dir", type=str, default=defaults.get("fixed_split_dir", None))
    parser.add_argument("--val_fraction", type=float, default=defaults.get("val_fraction", 0.1))
    parser.add_argument("--test_fraction", type=float, default=defaults.get("test_fraction", 0.1))
    parser.add_argument("--dataset_version", type=str, default=defaults.get("dataset_version", "pdbbind2020_local"))
    parser.add_argument("--experiments_root", type=str, default=defaults.get("experiments_root", "experiments"))
    parser.add_argument("--benchmark_dir", type=str, default=None)
    parser.add_argument("--python_executable", type=str, default=sys.executable)
    parser.add_argument("--dry_run", action="store_true", help="Write the benchmark plan without launching training.")
    parser.add_argument("--allow_duplicate", action="store_true", default=defaults.get("allow_duplicate", False))
    parser.add_argument("--log_level", type=str, default=defaults.get("log_level", "INFO"))
    
    # Hardware-aware parameters
    parser.add_argument("--num_workers", type=int, default=defaults.get("num_workers", 0))
    parser.add_argument("--pin_memory", action="store_true", default=defaults.get("pin_memory", False))
    parser.add_argument("--benchmark_version", type=str, default=defaults.get("benchmark_version", "Validated Baseline v0.95"))
    parser.add_argument("--hidden_dim", type=int, default=defaults.get("hidden_dim", 256))
    parser.add_argument("--dropout", type=float, default=defaults.get("dropout", 0.2))
    parser.add_argument("--optimizer", type=str, default=defaults.get("optimizer", "adam"), choices=["adam", "adamw", "radam"])
    parser.add_argument("--scheduler", type=str, default=defaults.get("scheduler", "none"), choices=["none", "plateau", "cosine", "onecycle"])
    parser.add_argument("--grad_clip", type=float, default=defaults.get("grad_clip", None))
    parser.add_argument("--no_residual", action="store_true", default=defaults.get("no_residual", False))
    parser.add_argument("--no_layernorm", action="store_true", default=defaults.get("no_layernorm", False))
    parser.add_argument("--no_attention", action="store_true", default=defaults.get("no_attention", False))
    parser.add_argument("--pooling", type=str, default=defaults.get("pooling", "mean"), choices=["mean", "max"])
    parser.add_argument("--model_type", type=str, default=defaults.get("model_type", "baseline"), choices=["baseline", "geometry", "geometry_attention", "geometry_interaction", "physics_guided", "soft_routing", "foundation_protein", "foundation_protein_only", "foundation_ligand", "foundation_ligand_only", "foundation_hybrid", "foundation_hybrid_only", "equivariant", "equivariant_only", "uncertainty"])
    parser.add_argument("--equivariant_model", type=str, default=defaults.get("equivariant_model", "egnn"), choices=["egnn"])
    parser.add_argument("--equiv_layers", type=int, default=defaults.get("equiv_layers", 4))
    parser.add_argument("--equiv_edge_dim", type=int, default=defaults.get("equiv_edge_dim", 32))
    parser.add_argument("--fusion", type=str, default=defaults.get("fusion", "cross_attention"), choices=["cross_attention", "concatenation", "weighted_sum", "gated", "residual", "no_fusion"])
    parser.add_argument("--protein_model", type=str, default=defaults.get("protein_model", "esm2"), choices=["esm2", "prot_t5", "none"])
    parser.add_argument("--ligand_model", type=str, default=defaults.get("ligand_model", "chemberta"), choices=["chemberta", "molformer", "none"])
    parser.add_argument("--layer_selection", type=str, default=defaults.get("layer_selection", "last"), choices=["last", "last4"])
    parser.add_argument("--pooling_strategy", type=str, default=defaults.get("pooling_strategy", "pocket_only"), choices=["pocket_only", "mean", "cls"])
    parser.add_argument("--ligand_pooling_strategy", type=str, default=defaults.get("ligand_pooling_strategy", "graph_node_mapping"), choices=["graph_node_mapping", "mean", "cls"])
    parser.add_argument("--train_fraction", type=float, default=defaults.get("train_fraction", 1.0))
    parser.add_argument("--router_type", type=str, default=defaults.get("router_type", "gated"), choices=["linear", "residual", "gated"])
    parser.add_argument("--routing_level", type=str, default=defaults.get("routing_level", "embedding"), choices=["embedding", "attention_head"])
    parser.add_argument("--gate_regularization", type=float, default=defaults.get("gate_regularization", 0.0))
    parser.add_argument("--entropy_regularization", type=float, default=defaults.get("entropy_regularization", 0.01))
    parser.add_argument("--sparsity_regularization", type=float, default=defaults.get("sparsity_regularization", 0.0))
    parser.add_argument("--diversity_regularization", type=float, default=defaults.get("diversity_regularization", 0.0))
    parser.add_argument("--router_capacity", type=str, default=defaults.get("router_capacity", "base"), choices=["tiny", "small", "base"])
    parser.add_argument("--contact_threshold", type=float, default=defaults.get("contact_threshold", 4.0))
    parser.add_argument("--loss_weight", type=float, default=defaults.get("loss_weight", 0.1))
    parser.add_argument("--loss_weight_strategy", type=str, default=defaults.get("loss_weight_strategy", "fixed"), choices=["fixed", "learnable", "cosine", "linear"])
    parser.add_argument("--contact_hidden_dim", type=int, default=defaults.get("contact_hidden_dim", 128))
    parser.add_argument("--rbf_basis", type=int, default=defaults.get("rbf_basis", 32))
    parser.add_argument("--rbf_start", type=float, default=defaults.get("rbf_start", 0.0))
    parser.add_argument("--rbf_stop", type=float, default=defaults.get("rbf_stop", 12.0))
    parser.add_argument("--rbf_proj_dim", type=int, default=defaults.get("rbf_proj_dim", 32))
    parser.add_argument("--interaction_dim", type=int, default=defaults.get("interaction_dim", 32))
    parser.add_argument("--latent_dim", type=int, default=defaults.get("latent_dim", 32))
    parser.add_argument("--chemistry_dim", type=int, default=defaults.get("chemistry_dim", 32))
    parser.add_argument("--representation_dim", type=int, default=defaults.get("representation_dim", 32))
    parser.add_argument("--attention_bias_type", type=str, default=defaults.get("attention_bias_type", "rbf"), choices=["rbf", "learned"])
    parser.add_argument("--attention_learnable_scale", type=str, default=defaults.get("attention_learnable_scale", "true"), choices=["true", "false"])
    parser.add_argument("--attention_initial_alpha", type=float, default=defaults.get("attention_initial_alpha", 0.1))
    parser.add_argument("--attention_normalize_bias", type=str, default=defaults.get("attention_normalize_bias", "true"), choices=["true", "false"])
    
    # Uncertainty configuration arguments
    parser.add_argument("--uncertainty_method", type=str, default=defaults.get("uncertainty_method", "mc_dropout"), choices=["mc_dropout", "ensemble", "evidential"])
    parser.add_argument("--mc_samples", type=int, default=defaults.get("mc_samples", 30))
    parser.add_argument("--ensemble_size", type=int, default=defaults.get("ensemble_size", 5))
    parser.add_argument("--calibration_method", type=str, default=defaults.get("calibration_method", "none"), choices=["none", "temperature_scaling", "variance_scaling"])
    parser.add_argument("--temperature", type=float, default=defaults.get("temperature", 1.0))
    parser.add_argument("--variance_multiplier", type=float, default=defaults.get("variance_multiplier", 1.0))

    return parser.parse_args(remaining_args)


def build_train_command(args, seed: int, experiment_dir: Path) -> List[str]:
    save_path = args.save_path or str(experiment_dir / "checkpoint.pt")
    command = [
        args.python_executable,
        "protliggnn_train.py",
        "--data_dir",
        args.data_dir,
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--patience",
        str(args.patience),
        "--seed",
        str(seed),
        "--run_name",
        f"{args.run_name}_seed_{seed}",
        "--save_path",
        save_path,
        "--split_strategy",
        args.split_strategy,
        "--split_dir",
        args.split_dir,
        "--val_fraction",
        str(args.val_fraction),
        "--test_fraction",
        str(args.test_fraction),
        "--dataset_version",
        args.dataset_version,
        "--experiments_root",
        args.experiments_root,
        "--experiment_dir",
        str(experiment_dir),
        "--log_level",
        args.log_level,
        "--num_workers",
        str(args.num_workers),
        "--research_version",
        args.benchmark_version,
    ]
    if args.max_samples is not None:
        command.extend(["--max_samples", str(args.max_samples)])
    if args.fixed_split_dir:
        command.extend(["--fixed_split_dir", args.fixed_split_dir])
    if args.no_crossgraph:
        command.append("--no_crossgraph")
    if args.allow_duplicate:
        command.append("--allow_duplicate")
    if args.pin_memory:
        command.append("--pin_memory")
        
    # Forward HPO hyperparameters
    command.extend(["--hidden_dim", str(args.hidden_dim)])
    command.extend(["--dropout", str(args.dropout)])
    command.extend(["--optimizer", args.optimizer])
    command.extend(["--scheduler", args.scheduler])
    command.extend(["--pooling", args.pooling])
    if args.grad_clip is not None:
        command.extend(["--grad_clip", str(args.grad_clip)])
    if args.no_residual:
        command.append("--no_residual")
    if args.no_layernorm:
        command.append("--no_layernorm")
    if args.no_attention:
        command.append("--no_attention")
        
    # Forward geometry parameters
    command.extend(["--model_type", args.model_type])
    command.extend(["--rbf_basis", str(args.rbf_basis)])
    command.extend(["--rbf_start", str(args.rbf_start)])
    command.extend(["--rbf_stop", str(args.rbf_stop)])
    command.extend(["--rbf_proj_dim", str(args.rbf_proj_dim)])
    command.extend(["--interaction_dim", str(args.interaction_dim)])
    command.extend(["--latent_dim", str(args.latent_dim)])
    command.extend(["--chemistry_dim", str(args.chemistry_dim)])
    command.extend(["--representation_dim", str(args.representation_dim)])
    command.extend(["--attention_bias_type", args.attention_bias_type])
    command.extend(["--attention_learnable_scale", args.attention_learnable_scale])
    command.extend(["--attention_initial_alpha", str(args.attention_initial_alpha)])
    command.extend(["--attention_normalize_bias", args.attention_normalize_bias])
    
    # Forward physics guided parameters
    command.extend(["--contact_threshold", str(args.contact_threshold)])
    command.extend(["--loss_weight", str(args.loss_weight)])
    command.extend(["--loss_weight_strategy", args.loss_weight_strategy])
    command.extend(["--contact_hidden_dim", str(args.contact_hidden_dim)])
    
    # Forward soft routing parameters
    command.extend(["--router_type", args.router_type])
    command.extend(["--routing_level", args.routing_level])
    command.extend(["--gate_regularization", str(args.gate_regularization)])
    command.extend(["--entropy_regularization", str(args.entropy_regularization)])
    command.extend(["--sparsity_regularization", str(args.sparsity_regularization)])
    command.extend(["--diversity_regularization", str(args.diversity_regularization)])
    command.extend(["--router_capacity", args.router_capacity])
    
    # Forward foundation model parameters
    command.extend(["--protein_model", args.protein_model])
    command.extend(["--ligand_model", args.ligand_model])
    command.extend(["--layer_selection", args.layer_selection])
    command.extend(["--pooling_strategy", args.pooling_strategy])
    command.extend(["--ligand_pooling_strategy", args.ligand_pooling_strategy])
    command.extend(["--train_fraction", str(args.train_fraction)])
    
    # Forward uncertainty parameters
    command.extend(["--uncertainty_method", args.uncertainty_method])
    command.extend(["--mc_samples", str(args.mc_samples)])
    command.extend(["--ensemble_size", str(args.ensemble_size)])
    command.extend(["--calibration_method", args.calibration_method])
    command.extend(["--temperature", str(args.temperature)])
    command.extend(["--variance_multiplier", str(args.variance_multiplier)])
        
    return command


def get_gpu_stats() -> tuple:
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        parts = res.stdout.strip().split(",")
        gpu_util = float(parts[0].strip())
        gpu_mem_used = float(parts[1].strip())
        gpu_mem_total = float(parts[2].strip())
        return gpu_util, gpu_mem_used, gpu_mem_total
    except Exception:
        return 0.0, 0.0, 0.0


def draw_dashboard(state: Dict[str, Any], active_seed: Optional[int], active_dir: Optional[Path], epoch_stats: Dict[str, Any]):
    # Get system stats
    cpu_util = psutil.cpu_percent()
    ram = psutil.virtual_memory()
    ram_util = ram.percent
    ram_used_gb = ram.used / (1024**3)
    ram_total_gb = ram.total / (1024**3)
    gpu_util, gpu_mem_used, gpu_mem_total = get_gpu_stats()
    
    # ANSI code to clear screen & cursor home
    sys.stdout.write("\033[H\033[J")
    
    dashboard = []
    dashboard.append("================================================================================")
    dashboard.append("                       PROTLIGGNN BENCHMARK PROGRESS DASHBOARD                 ")
    dashboard.append("================================================================================")
    dashboard.append(f" Benchmark Version : {state.get('benchmark_version')}")
    dashboard.append(f" Configuration     : epochs={state.get('epochs')}, batch_size={epoch_stats.get('batch_size', 8)}, max_samples={state.get('max_samples')}, device={epoch_stats.get('device', 'cuda')}")
    dashboard.append(f" Benchmark Dir     : {state.get('benchmark_dir')}")
    dashboard.append("--------------------------------------------------------------------------------")
    dashboard.append("Seeds Status:")
    for s in state.get("seeds", []):
        run_info = state["runs"].get(str(s), {})
        status = run_info.get("status", "pending")
        exp_dir = run_info.get("experiment_dir", "")
        if status == "completed":
            metrics = run_info.get("metrics", {})
            rmse = metrics.get("rmse", "N/A")
            dashboard.append(f"  Seed {s:<5} : [Completed] (Dir: {exp_dir}, Val RMSE: {rmse})")
        elif status == "running":
            dashboard.append(f"  Seed {s:<5} : [Running  ] (Dir: {exp_dir})")
        elif status == "interrupted":
            dashboard.append(f"  Seed {s:<5} : [Interrupted] (Dir: {exp_dir})")
        else:
            dashboard.append(f"  Seed {s:<5} : [Pending  ]")
            
    dashboard.append("--------------------------------------------------------------------------------")
    if active_seed is not None:
        dashboard.append(f"ACTIVE RUN METRICS (Seed {active_seed} - {active_dir}):")
        dashboard.append(f"  Epoch           : {epoch_stats.get('epoch', 0)} / {state.get('epochs')}")
        train_loss = epoch_stats.get('train_loss')
        train_loss_str = f"{train_loss:.4f}" if train_loss is not None else "N/A"
        dashboard.append(f"  Training Loss   : {train_loss_str}")
        val_rmse = epoch_stats.get('val_rmse')
        val_rmse_str = f"{val_rmse:.4f}" if val_rmse is not None else "N/A"
        dashboard.append(f"  Validation RMSE : {val_rmse_str}")
        val_mae = epoch_stats.get('val_mae')
        val_mae_str = f"{val_mae:.4f}" if val_mae is not None else "N/A"
        dashboard.append(f"  Validation MAE  : {val_mae_str}")
        dashboard.append(f"  Learning Rate   : {epoch_stats.get('lr', 'N/A')}")
        epoch_time = epoch_stats.get('epoch_time')
        epoch_time_str = f"{epoch_time:.2f}s" if epoch_time is not None else "N/A"
        dashboard.append(f"  Time per Epoch  : {epoch_time_str}")
        dashboard.append(f"  Est. Remaining  : {epoch_stats.get('eta', 'Estimating...')}")
        dashboard.append(f"  Checkpoint      : {epoch_stats.get('checkpoint_status', 'None')}")
        dashboard.append(f"  Early Stopping  : Counter: {epoch_stats.get('early_stopping', '0 / 10')}")
    else:
        dashboard.append("ACTIVE RUN METRICS : None")
        
    dashboard.append("--------------------------------------------------------------------------------")
    dashboard.append("SYSTEM RESOURCES:")
    dashboard.append(f"  CPU Utilization : {cpu_util:.1f}%")
    dashboard.append(f"  RAM Utilization : {ram_util:.1f}% ({ram_used_gb:.1f} GB / {ram_total_gb:.1f} GB)")
    if gpu_mem_total > 0:
        dashboard.append(f"  GPU Utilization : {gpu_util:.1f}%")
        dashboard.append(f"  GPU VRAM Usage  : {gpu_mem_used:.1f} MB / {gpu_mem_total:.1f} MB")
    else:
        dashboard.append("  GPU Utilization : N/A (CPU execution mode)")
    dashboard.append("================================================================================")
    
    dashboard_text = "\n".join(dashboard)
    sys.stdout.write(dashboard_text + "\n")
    sys.stdout.flush()


def update_dashboard_md(state: Dict[str, Any], artifact_dir: Optional[Path] = None):
    # Generates a rendered markdown dashboard summary
    lines = []
    lines.append(f"# ProtLigGNN Benchmark Dashboard ({state.get('benchmark_version')})")
    lines.append(f"- **Benchmark Directory:** `{state.get('benchmark_dir')}`")
    lines.append(f"- **Max Samples:** `{state.get('max_samples')}`")
    lines.append(f"- **Epochs:** `{state.get('epochs')}`")
    lines.append(f"- **Device:** `{state.get('runs', {}).get(str(state['seeds'][0]), {}).get('device', 'cuda')}`")
    lines.append("\n## Seed Execution Progress\n")
    lines.append("| Seed | Directory | Status | Best Val RMSE | Best Val MAE | Completed UTC |")
    lines.append("|---|---|---|---|---|---|")
    
    for s in state.get("seeds", []):
        run_info = state["runs"].get(str(s), {})
        status = run_info.get("status", "pending")
        exp_dir = run_info.get("experiment_dir", "")
        metrics = run_info.get("metrics", {})
        rmse = f"{metrics.get('rmse', 'N/A')}"
        mae = f"{metrics.get('mae', 'N/A')}"
        comp_utc = run_info.get("completed_utc", "N/A")
        lines.append(f"| {s} | `{exp_dir}` | {status} | {rmse} | {mae} | {comp_utc} |")
        
    dashboard_md_text = "\n".join(lines)
    
    # Save inside benchmark dir
    b_dir = Path(state["benchmark_dir"])
    b_dir.mkdir(parents=True, exist_ok=True)
    (b_dir / "benchmark_dashboard.md").write_text(dashboard_md_text, encoding="utf-8")
    
    # Also save to the artifacts directory so the user can easily view it in real-time
    if artifact_dir and artifact_dir.exists():
        (artifact_dir / "benchmark_dashboard.md").write_text(dashboard_md_text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    seeds = parse_seed_argument(args.seeds)
    experiments_root = Path(args.experiments_root)
    experiments_root.mkdir(parents=True, exist_ok=True)
    
    # Define artifact path for the workspace
    artifact_dir = Path("C:/Users/Harshhhh/.gemini/antigravity/brain/03c21182-9e56-4764-a6fd-83660d6d0fc1")
    
    # Check for active benchmark progress to resume
    progress_path = Path("experiments/benchmark_progress.json")
    state = {}
    is_resuming = False
    
    if progress_path.exists():
        try:
            loaded = read_json(progress_path)
            # Verify compatibility of configurations
            if (loaded.get("no_crossgraph") == args.no_crossgraph and
                loaded.get("max_samples") == args.max_samples and
                loaded.get("epochs") == args.epochs and
                loaded.get("benchmark_version") == args.benchmark_version):
                state = loaded
                is_resuming = True
        except Exception as e:
            # Silently fallback to fresh plan
            pass
            
    if not is_resuming:
        benchmark_dir = Path(args.benchmark_dir) if args.benchmark_dir else experiments_root / f"benchmark_{utc_now().replace(':', '').replace('+', '_')}"
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "benchmark_version": args.benchmark_version,
            "benchmark_dir": str(benchmark_dir),
            "no_crossgraph": args.no_crossgraph,
            "max_samples": args.max_samples,
            "epochs": args.epochs,
            "seeds": seeds,
            "runs": {}
        }
        # Pre-initialize runs
        for s in seeds:
            state["runs"][str(s)] = {"status": "pending"}
        write_json(progress_path, state)
    else:
        benchmark_dir = Path(state["benchmark_dir"])
        benchmark_dir.mkdir(parents=True, exist_ok=True)

    run_rows = []
    plan = {
        "created_utc": utc_now(),
        "seeds": seeds,
        "benchmark_dir": str(benchmark_dir),
        "args": vars(args),
        "runs": [],
    }
    write_json(benchmark_dir / "benchmark_plan.json", plan)

    # Initialize live log
    live_log_path = Path("benchmark_live.log")
    if not is_resuming:
        live_log_path.write_text(f"=== Benchmark Started: {utc_now()} ===\n", encoding="utf-8")
    else:
        with live_log_path.open("a", encoding="utf-8") as lf:
            lf.write(f"=== Benchmark Resumed: {utc_now()} ===\n")

    for seed in seeds:
        run_info = state["runs"].setdefault(str(seed), {"status": "pending"})
        status = run_info.get("status", "pending")
        
        # Check if already completed and verify metrics.json exists
        experiment_dir = Path(run_info.get("experiment_dir", "")) if run_info.get("experiment_dir") else resolve_experiment_dir(experiments_root)
        run_info["experiment_dir"] = str(experiment_dir)
        metrics_path = experiment_dir / "metrics.json"
        
        if status == "completed" and metrics_path.exists():
            # Load metrics without executing
            try:
                metrics = read_json(metrics_path)
                row = {"seed": seed, "run_dir": str(experiment_dir)}
                row.update(metrics)
                run_rows.append(row)
                continue
            except Exception:
                # If reading metrics failed, force retraining
                pass

        # Prepare execution mode: check for checkpoint presence to enable resume
        checkpoint_path = experiment_dir / "checkpoint.pt"
        resume_mode = checkpoint_path.exists()
        
        # Update progress state to running
        run_info["status"] = "running"
        run_info["device"] = args.device
        write_json(progress_path, state)
        
        command = build_train_command(args, seed, experiment_dir)
        if resume_mode:
            # We must resume from the existing checkpoint
            # Check if command already contains --resume. If not, append it.
            if "--resume" not in command:
                command.extend(["--resume", str(experiment_dir.name)])
                
        command_text = " ".join(command)
        plan["runs"].append({"seed": seed, "experiment_dir": str(experiment_dir), "command": command})
        write_json(benchmark_dir / "benchmark_plan.json", plan)

        if args.dry_run:
            print(f"[dry-run] {command_text}")
            run_info["status"] = "dry_run"
            write_json(progress_path, state)
            continue

        log_path = experiment_dir / "benchmark_subprocess.log"
        
        # Initialize active epoch stats
        epoch_stats = {
            "epoch": 0,
            "train_loss": None,
            "val_rmse": None,
            "val_mae": None,
            "lr": args.lr,
            "epoch_time": None,
            "best_rmse": None,
            "best_epoch": None,
            "early_stopping": "0 / 10",
            "checkpoint_status": "None",
            "device": args.device,
            "batch_size": args.batch_size
        }
        
        epoch_start_time = time.time()
        
        # Start training subprocess
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream logs in real-time
        with log_path.open("w", encoding="utf-8") as sub_log, live_log_path.open("a", encoding="utf-8") as live_log:
            sub_log.write(f"Command: {command_text}\n\n")
            
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if not line:
                    continue
                
                sub_log.write(line)
                sub_log.flush()
                
                line_stripped = line.strip()
                
                # Parse metrics & events
                if "train loss=" in line_stripped:
                    # Epoch 001 train loss=32.4284 PCC=-0.5022 Spearman=-0.5000 RMSE=5.6946 MAE=5.6172 R2=-37.7850
                    match = re.search(
                        r"Epoch (\d+) train loss=([\d\.\-]+) PCC=([\d\.\-]+) Spearman=([\d\.\-]+) RMSE=([\d\.\-]+) MAE=([\d\.\-]+)",
                        line_stripped
                    )
                    if match:
                        epoch_stats["epoch"] = int(match.group(1))
                        epoch_stats["train_loss"] = float(match.group(2))
                        epoch_stats["epoch_time"] = time.time() - epoch_start_time
                        epoch_start_time = time.time()
                        
                elif "val loss=" in line_stripped:
                    # Epoch 001 val loss=9.5388 PCC=nan Spearman=nan RMSE=3.0885 MAE=3.0885 R2=nan
                    match = re.search(
                        r"Epoch (\d+) val loss=([\d\.\-]+) PCC=([\d\.\-]+) Spearman=([\d\.\-]+) RMSE=([\d\.\-]+) MAE=([\d\.\-]+)",
                        line_stripped
                    )
                    if match:
                        epoch_stats["val_rmse"] = float(match.group(5))
                        epoch_stats["val_mae"] = float(match.group(6))
                        
                elif "Saved checkpoint for epoch" in line_stripped:
                    # Saved checkpoint for epoch 1 with validation RMSE 3.0885008573532104.
                    match = re.search(
                        r"Saved checkpoint for epoch (\d+) with validation RMSE ([\d\.\-]+)",
                        line_stripped
                    )
                    if match:
                        c_epoch = int(match.group(1))
                        c_rmse = float(match.group(2).rstrip('.'))
                        epoch_stats["best_rmse"] = c_rmse
                        epoch_stats["best_epoch"] = c_epoch
                        epoch_stats["checkpoint_status"] = f"Saved (Best RMSE: {c_rmse:.4f})"
                        epoch_stats["early_stopping"] = f"0 / {args.patience}"
                        
                elif "Early stopping triggered" in line_stripped:
                    epoch_stats["early_stopping"] = "Triggered"
                    
                elif "Resumed from epoch" in line_stripped:
                    # Resumed from epoch 1; continuing at epoch 2.
                    match = re.search(r"Resumed from epoch (\d+); continuing at epoch (\d+)", line_stripped)
                    if match:
                        epoch_stats["epoch"] = int(match.group(2)) - 1
                        
                # Estimate remaining time
                if epoch_stats["epoch"] > 0 and epoch_stats["epoch_time"] is not None:
                    rem_epochs = max(0, args.epochs - epoch_stats["epoch"])
                    est_rem = rem_epochs * epoch_stats["epoch_time"]
                    epoch_stats["eta"] = f"{int(est_rem)}s"
                
                # Draw live dashboard in the terminal
                draw_dashboard(state, seed, experiment_dir, epoch_stats)
                
                # Append progress info to live log
                if "train loss=" in line_stripped:
                    log_entry = f"[{utc_now()}] Seed: {seed} | Epoch: {epoch_stats['epoch']}/{args.epochs} | Train Loss: {epoch_stats.get('train_loss')} | Val RMSE: {epoch_stats.get('val_rmse')} | ETA: {epoch_stats.get('eta')}\n"
                    live_log.write(log_entry)
                    live_log.flush()
        
        # Verify exit status of subprocess
        if process.returncode != 0:
            run_info["status"] = "interrupted"
            write_json(progress_path, state)
            
            failure = {
                "seed": seed,
                "experiment_dir": str(experiment_dir),
                "returncode": process.returncode,
                "log": str(log_path),
            }
            write_json(benchmark_dir / "benchmark_failure.json", failure)
            print(f"Benchmark failed for seed {seed}. See {log_path}")
            return process.returncode

        # Post-run metrics check
        if not metrics_path.exists():
            run_info["status"] = "failed"
            write_json(progress_path, state)
            
            failure = {
                "seed": seed,
                "experiment_dir": str(experiment_dir),
                "error": "metrics.json was not produced",
                "log": str(log_path),
            }
            write_json(benchmark_dir / "benchmark_failure.json", failure)
            print(f"Benchmark failed for seed {seed}: metrics.json was not produced.")
            return 1

        # Read completed metrics and update status
        metrics = read_json(metrics_path)
        run_info["status"] = "completed"
        run_info["metrics"] = metrics
        run_info["completed_utc"] = utc_now()
        write_json(progress_path, state)
        
        row = {"seed": seed, "run_dir": str(experiment_dir)}
        row.update(metrics)
        run_rows.append(row)
        
        # Update dashboard markdown report
        update_dashboard_md(state, artifact_dir)
        
        write_json(benchmark_dir / "completed_runs.json", {"runs": run_rows})

    if args.dry_run:
        print(f"Benchmark dry run plan written to {benchmark_dir / 'benchmark_plan.json'}")
        return 0

    # Summary reporting
    summary = write_results_summary(run_rows, benchmark_dir, benchmark_name=args.run_name)
    print(f"Benchmark complete. Report: {summary['benchmark_report']}")
    
    # Generate standardized experiment package (reports, metrics, checkpoints, figures, statistics, environment, logs)
    try:
        from artifact_framework import generate_benchmark_artifacts
        generate_benchmark_artifacts(benchmark_dir, run_rows, args)
    except Exception as e:
        print(f"Warning: Failed to generate standardized experiment artifacts: {e}")
        import traceback
        traceback.print_exc()
        
    # Delete progress tracking file on successful completion of the entire suite
    if progress_path.exists():
        progress_path.unlink()
        
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
