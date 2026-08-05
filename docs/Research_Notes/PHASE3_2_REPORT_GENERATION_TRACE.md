# Phase 3.2 Report Generation Trace
**Date**: 2026-07-16  

---

### 1. Data Processing Pipeline
The following diagram traces how each metrics file is propagated through the post-processing framework:
```mermaid
graph TD
    A[Raw Model Predictions contact_predictions.pt] --> B[Metrics Evaluator: contact_metrics.json]
    B --> C[artifact_framework.py: get_reports_markdown]
    C --> D[12_Contact_Analysis_Report.md]
    D --> E[Chrome headless PDF: interaction_analysis_report.pdf]
```

---

### 2. Stale Path Diagnosis
- **The Issue**: `artifact_framework.py` originally hardcoded static template values for the contact analysis section instead of querying `contact_metrics.json`.
- **The Solution**: Modified `get_reports_markdown` to dynamically load and format contact metrics and multitask history from the benchmark folder. Reports 12, 13, and 14 are now 100% dynamic.
