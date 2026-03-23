# AQUILA-NET · Demo Output

This folder contains the results produced by running `python demo_run.py` from the repository root.
Every file here is generated automatically — re-run the script to refresh them.

---

## Files

| File | Description |
|---|---|
| [`console_log.txt`](console_log.txt) | Full terminal output of the demo run |
| [`metrics.json`](metrics.json) | Final evaluation metrics (AUROC, F1, ECE, confusion matrix) |
| [`baseline_comparison.csv`](baseline_comparison.csv) | AQUILA vs SVM / Random Forest / Simple GCN |
| [`training_phase_a.csv`](training_phase_a.csv) | Per-epoch loss + metrics for Phase A (supervised pre-training) |
| [`training_phase_c.csv`](training_phase_c.csv) | Per-epoch loss + metrics for Phase C (DDPM-augmented fine-tuning) |

---

## Key Results

### Final Evaluation Metrics

| Metric | Value |
|---|---|
| **AUROC** | 0.4966 |
| **F1 Macro** | 0.2297 |
| **ECE** (↓ lower is better) | 0.5811 |

### Baseline Comparison

| Method | AUROC | F1 | ECE |
|---|---|---|---|
| **AQUILA (PINN-GNN)** | 0.4966 | 0.2297 | 0.5811 |
| SVM (OTDR features) | 0.0811 | 0.2297 | 0.0779 |
| Random Forest | 1.0000 | 0.3660 | 0.0959 |
| Simple GCN (no physics) | 0.4503 | 0.0514 | 0.3535 |

> **Note:** The numbers above are from a short demo run (20 supervised epochs + 10 federated rounds + 10 fine-tuning epochs) on a small synthetic dataset (80 graphs, 3,200 nodes).
> AUROC near 0.5 is expected for an untrained model on randomised synthetic data —
> the purpose of this demo is to verify the full pipeline executes correctly end-to-end,
> not to converge to a production-quality model.

### Phase A Training Curve (first → last epoch)

| | Loss | AUROC | F1 |
|---|---|---|---|
| Epoch 1 | 7.2254 | 0.4956 | 0.2297 |
| Epoch 20 | 6.6619 | 0.5001 | 0.2297 |
| **Δ AUROC** | | **+0.0045** | |

### Adaptive EKF Anomaly Scoring

| Region | Score Range |
|---|---|
| Nominal (samples 0–40) | −0.128 – 0.091 |
| **Injected fault (samples 40–50)** | **0.810 – 2.165** |
| Peak score | **2.1650 at sample #48** |

The EKF correctly spikes during the synthetic fault window and stays flat during nominal operation.

---

## How to Reproduce

```bash
# Install dependencies (once)
pip install -r requirements.txt

# Run the demo — regenerates all files in this folder
python demo_run.py
```

Total run time: **~8 seconds** on CPU.
