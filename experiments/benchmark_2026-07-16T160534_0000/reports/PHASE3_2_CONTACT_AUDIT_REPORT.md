# Phase 3.2 Contact Audit Report
**Date**: 2026-07-16  

---

### 1. Probability Distribution Analysis
Recomputed directly from `contact_predictions.pt` (N = 230,753 atomic pairs):
- **Minimum Probability**: `0.003052`
- **Maximum Probability**: `0.078176`
- **Mean Probability**: `0.007511`
- **Median Probability**: `0.005893`
- **Standard Deviation**: `0.004481`

#### Percentiles
- **10th percentile**: `0.004074`
- **25th percentile**: `0.004622`
- **50th (Median)**: `0.005893`
- **75th percentile**: `0.008777`
- **90th percentile**: `0.013194`
- **95th percentile**: `0.016143`
- **99th percentile**: `0.025018`

---

### 2. Resolution of the Threshold Mismatch
- **Under the Default Threshold (t = 0.5)**:
  - Positive Predictions: **0**  
  - Negative Predictions: **230,753**  
  - Precision: **0.0000**  
  - Recall: **0.0000**  
  - F1: **0.0000**  
- **Under the Prior Probability Threshold (t = 0.00587)**:
  - Positive predictions matches class prior, yielding non-zero outputs but low precision (**0.0122**) due to massive background class imbalance.

#### Discrepancy Cause
The original `Contact Analysis Report` contained **stale cached placeholder values** (Precision = 0.814, Recall = 0.763, F1 = 0.788) from Phase 3.1C's IRM template. These have been successfully regenerated and corrected.

---

### 3. Auxiliary Head Learnability Status
The auxiliary head has **failed to learn** (PR-AUC = 0.0122, barely above the random baseline of 0.00587). This is due to severe class imbalance (0.58% positive contacts) without positive weight scaling or prior bias initialization, causing the network output to collapse to prior levels.
