# AQUILA-NET

**Physics-Aware Federated Graph Neural Network for Undersea Cable Anomaly Detection**

AQUILA-NET fuses real-world data streams (TeleGeography cable maps, GEBCO bathymetry, IRIS seismic catalogs, AIS vessel tracks, OTDR waveforms, ICPC fault reports) with a physics-informed GNN trained under differential-privacy constraints to provide near-real-time fault attribution for the global submarine cable network.

---

## Architecture

```
Real-World Data           Physics-Informed GNN           Federated Learning
─────────────────         ─────────────────────           ─────────────────
TeleGeography API    ──►  PINN-GNN (Euler-Bernoulli /     Opacus DP-SGD
GEBCO NetCDF         ──►  Telegrapher PDE residuals)  ──► FedAvg aggregation
IRIS Seismic FDSN    ──►  AR(2)-EKF anomaly scores        Async staleness tol.
AIS Vessel tracks    ──►  DDPM waveform augmentation  ──► Privacy accounting
OTDR physics sim     ──►  GNNExplainer attribution
ICPC fault catalog   ──►
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt          # or: make install

# 2. Fetch real-world data
python data/dataset_builder.py --fetch-real   # or: make fetch

# 3. Train
python training/train_aquila.py              # or: make train

# 4. Launch dashboard (Dash on http://localhost:8050)
python dashboard/app.py                       # or: make dashboard

# 5. Run tests
pytest tests/ -v                              # or: make test
```

## Module Overview

| Module | Description |
|---|---|
| `data/` | TeleGeography, GEBCO, IRIS, AIS, OTDR, ICPC fetchers |
| `graph/` | PINN-GNN layers, message passing, edge features, topology |
| `models/` | Encoder, anomaly detector, attribution classifier, PDE residuals |
| `filters/` | Adaptive EKF with AR(2) prior and CUSUM change detection |
| `diffusion/` | DDPM-based waveform augmentation with cable conditioner |
| `federated/` | DP-SGD training, FedAvg, async handler, privacy accountant |
| `explainability/` | GNNExplainer, MC-Dropout uncertainty, calibration |
| `evaluation/` | AUROC, F1, ECE, confusion matrix, benchmark runner |
| `training/` | Trainer, multi-task losses, LR scheduler callbacks |
| `dashboard/` | 5-page Dash app (Ops Center, Training Monitor, Analysis Lab, …) |

## Tests

```
pytest tests/ -v
```

34 tests across 6 test files, all passing.

## Configuration

YAML configs live in `configs/`:
- `gnn_config.yaml` — GNN architecture
- `ekf_config.yaml` — EKF / CUSUM thresholds
- `diffusion_config.yaml` — DDPM schedule
- `federated_config.yaml` — FL rounds & DP budget
- `cable_config.yaml` — Cable material & physics constants
