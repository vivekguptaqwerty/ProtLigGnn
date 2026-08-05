import numpy as np
import scipy.stats as stats
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, auc, average_precision_score,
    balanced_accuracy_score, matthews_corrcoef, brier_score_loss
)
from typing import Dict

def compute_contact_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray) -> Dict[str, float]:
    """Computes standard binary classification metrics for predicted contact maps."""
    y_true_flat = y_true.flatten()
    y_pred_prob_flat = y_pred_prob.flatten()
    
    y_pred_binary = (y_pred_prob_flat >= 0.5).astype(float)
    
    try:
        roc_auc = roc_auc_score(y_true_flat, y_pred_prob_flat)
    except Exception:
        roc_auc = 0.5
        
    try:
        precision, recall, _ = precision_recall_curve(y_true_flat, y_pred_prob_flat)
        pr_auc = auc(recall, precision)
    except Exception:
        pr_auc = 0.0
        
    return {
        "precision": float(precision_score(y_true_flat, y_pred_binary, zero_division=0)),
        "recall": float(recall_score(y_true_flat, y_pred_binary, zero_division=0)),
        "f1": float(f1_score(y_true_flat, y_pred_binary, zero_division=0)),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "average_precision": float(average_precision_score(y_true_flat, y_pred_prob_flat)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_flat, y_pred_binary)),
        "mcc": float(matthews_corrcoef(y_true_flat, y_pred_binary)),
        "brier_score": float(brier_score_loss(y_true_flat, y_pred_prob_flat))
    }

def compute_localization_metrics(attention_weights: np.ndarray, distances: np.ndarray, threshold: float = 4.0) -> Dict[str, float]:
    """Computes spatial attention alignment and localization metrics."""
    attn_flat = attention_weights.flatten()
    dist_flat = distances.flatten()
    
    try:
        dist_corr, _ = stats.pearsonr(attn_flat, dist_flat)
    except Exception:
        dist_corr = 0.0
        
    eps = 1e-12
    attn_norm = attn_flat / (np.sum(attn_flat) + eps)
    entropy = -np.sum(attn_norm * np.log(attn_norm + eps))
    
    true_contacts_mask = dist_flat < threshold
    num_contacts = int(np.sum(true_contacts_mask))
    
    if num_contacts == 0:
        contact_recall = 0.0
        topk_loc = 0.0
    else:
        topk_indices = np.argsort(attn_flat)[-num_contacts:]
        captured_contacts = np.sum(true_contacts_mask[topk_indices])
        contact_recall = float(captured_contacts) / num_contacts
        topk_loc = contact_recall
        
    return {
        "attention_entropy": float(entropy),
        "distance_correlation": float(dist_corr),
        "contact_recall": float(contact_recall),
        "topk_localization": float(topk_loc)
    }
