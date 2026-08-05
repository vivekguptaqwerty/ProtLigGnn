import torch
import numpy as np
from typing import Dict, Any, Tuple

def compute_linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Computes Linear Centered Kernel Alignment (CKA) between x and y."""
    x_centered = x - x.mean(dim=0, keepdim=True)
    y_centered = y - y.mean(dim=0, keepdim=True)
    
    # Covariance inner product
    num = torch.linalg.norm(torch.matmul(x_centered.t(), y_centered), ord='fro') ** 2
    den = torch.linalg.norm(torch.matmul(x_centered.t(), x_centered), ord='fro') * \
          torch.linalg.norm(torch.matmul(y_centered.t(), y_centered), ord='fro')
    
    if den < 1e-8:
        return 0.0
    return float(num / den)

def compute_svcca(x: torch.Tensor, y: torch.Tensor, threshold: float = 0.99) -> float:
    """Computes Singular Value Canonical Correlation Analysis (SVCCA) between x and y."""
    N, D = x.shape
    if N <= 1:
        return 1.0
        
    x_c = x - x.mean(dim=0, keepdim=True)
    y_c = y - y.mean(dim=0, keepdim=True)
    
    # Compute D x D covariance matrices
    cov_xx = torch.matmul(x_c.t(), x_c) / (N - 1)
    cov_yy = torch.matmul(y_c.t(), y_c) / (N - 1)
    cov_xy = torch.matmul(x_c.t(), y_c) / (N - 1)
    
    try:
        ux, sx, vx = torch.linalg.svd(cov_xx)
        uy, sy, vy = torch.linalg.svd(cov_yy)
        
        def get_k(s):
            cum = torch.cumsum(s, dim=0)
            tot = cum[-1]
            if tot < 1e-8:
                return 1
            cum = cum / tot
            k = torch.searchsorted(cum, threshold).item() + 1
            return max(1, min(k, D))
            
        kx = get_k(sx)
        ky = get_k(sy)
        
        # Projection directions
        Wx = ux[:, :kx] * torch.rsqrt(sx[:kx] + 1e-8)
        Wy = uy[:, :ky] * torch.rsqrt(sy[:ky] + 1e-8)
        
        # Canonical covariance
        T = torch.matmul(torch.matmul(Wx.t(), cov_xy), Wy)
        _, sv, _ = torch.linalg.svd(T)
        
        return float(torch.mean(sv))
    except Exception:
        # Fallback to mean cosine similarity on fail
        cos = torch.nn.functional.cosine_similarity(x_c, y_c, dim=1)
        return float(cos.mean())

def compute_gate_diagnostics(gate: torch.Tensor) -> Dict[str, float]:
    """Computes gating entropy, utilization, and sparsity metrics."""
    if gate is None:
        return {}
    
    mean_val = float(gate.mean())
    var_val = float(gate.var())
    
    # Entropy: -p log(p) - (1-p) log(1-p)
    eps = 1e-8
    p = torch.clamp(gate, eps, 1.0 - eps)
    entropy = - (p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
    mean_entropy = float(entropy.mean())
    
    # Sparsity: fraction of gates < 0.1
    sparsity = float((gate < 0.1).float().mean())
    # Utilization: fraction of gates > 0.9
    utilization = float((gate > 0.9).float().mean())
    
    return {
        "gate_mean": mean_val,
        "gate_variance": var_val,
        "gate_entropy": mean_entropy,
        "gate_sparsity": sparsity,
        "gate_utilization": utilization
    }

def compute_gradient_conflict(grad_aff: torch.Tensor, grad_contact: torch.Tensor) -> Dict[str, float]:
    """Computes gradient conflict metrics between affinity and contact prediction branches."""
    if grad_aff is None or grad_contact is None:
        return {}
        
    dot = float(torch.sum(grad_aff * grad_contact))
    norm_aff = float(torch.norm(grad_aff))
    norm_contact = float(torch.norm(grad_contact))
    
    den = norm_aff * norm_contact
    cos_sim = dot / den if den > 1e-8 else 0.0
    
    # Conflict is when dot product is negative
    conflict_magnitude = max(0.0, -dot)
    
    # PCGrad correction magnitude: if conflict, project grad_aff onto grad_contact's orthogonal plane
    pcgrad_proj = 0.0
    if dot < 0.0 and norm_contact > 1e-8:
        # projected vector = grad_aff - (dot / norm_contact^2) * grad_contact
        # projection correction magnitude is simply the norm of the projected term
        pcgrad_proj = abs(dot) / norm_contact
        
    return {
        "gradient_cosine_similarity": cos_sim,
        "gradient_conflict_frequency": 1.0 if dot < 0.0 else 0.0,
        "gradient_conflict_magnitude": conflict_magnitude,
        "pcgrad_correction_magnitude": pcgrad_proj,
        "task_agreement_score": max(0.0, cos_sim),
        "task_interference_score": max(0.0, -cos_sim)
    }
