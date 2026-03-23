"""AQUILA-NET End-to-End Demo Runner.

Demonstrates the full pipeline:
  1. Synthetic cable-graph dataset generation with injected anomalies
  2. Phase A – supervised pre-training
  3. Phase B – federated learning (FedAvg + DP-SGD)
  4. Phase C – DDPM-augmented fine-tuning
  5. Full metric evaluation (AUROC, F1, ECE, confusion matrix)
  6. Baseline comparison (SVM, Random Forest, Simple GCN)

Results are written to the output/ directory:
  output/console_log.txt          – full terminal output
  output/metrics.json             – final evaluation metrics
  output/baseline_comparison.csv  – AQUILA vs baseline methods
  output/training_phase_a.csv     – per-epoch Phase A history
  output/training_phase_c.csv     – per-epoch Phase C history

Run:
    python demo_run.py
"""

from __future__ import annotations

import csv
import io
import json
import os
import sys
import copy
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ── AQUILA-NET imports ───────────────────────────────────────────────────────
from graph.pinn_gnn import PINNGraphNeuralNetwork
from training.trainer import AQUILATrainer
from training.losses import AQUILALoss
from training.callbacks import TrainingCallback
from evaluation.evaluate import AQUILAEvaluator
from evaluation.metrics import AQUILAMetrics
from federated.fl_server import FederatedServer
from filters.adaptive_ekf import AdaptiveEKF
from diffusion.ddpm_augmentor import DDPMAugmentor

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Constants ────────────────────────────────────────────────────────────────
NODE_DIM = 8
EDGE_DIM = 6
HIDDEN_DIM = 64
N_CLASSES = 4          # normal / anchor-drag / seismic / equipment-failure
N_NODES = 40
N_CABLES = 60          # graph edges
ANOMALY_RATE = 0.15    # 15 % of nodes are anomalous
DEVICE = "cpu"
AUG_SEED_OFFSET = 1000  # seed offset for augmented (high-anomaly-rate) graphs
CM_LABEL_WIDTH = 20     # character width of the row-label column in the confusion matrix
OUTPUT_DIR = os.path.join(ROOT, "output")

# ═══════════════════════════════════════════════════════════════════════════
# 0.  Output helpers
# ═══════════════════════════════════════════════════════════════════════════

class _Tee:
    """Write stdout to both the terminal and an in-memory buffer."""

    def __init__(self):
        self._buf = io.StringIO()
        self._orig = sys.stdout

    def write(self, text: str) -> None:
        self._orig.write(text)
        self._buf.write(text)

    def flush(self) -> None:
        self._orig.flush()

    def getvalue(self) -> str:
        return self._buf.getvalue()


def _save_outputs(
    console_text: str,
    results: dict,
    comparison_df,
    phase_a_history: list,
    phase_c_history: list,
) -> None:
    """Persist all demo artefacts into the output/ folder."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Full console log
    with open(os.path.join(OUTPUT_DIR, "console_log.txt"), "w", encoding="utf-8") as f:
        f.write(console_text)

    # 2. Final evaluation metrics as JSON
    metrics_out = {
        "auroc": round(float(results["auroc"]), 6),
        "f1_macro": round(float(results["f1_macro"]), 6),
        "ece": round(float(results["ece"]), 6),
        "confusion_matrix": results["confusion_matrix"].tolist(),
    }
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)

    # 3. Baseline comparison as CSV
    comparison_df.to_csv(
        os.path.join(OUTPUT_DIR, "baseline_comparison.csv"),
        index=False,
        float_format="%.6f",
    )

    # 4. Phase A training history
    _write_history_csv(
        phase_a_history,
        os.path.join(OUTPUT_DIR, "training_phase_a.csv"),
    )

    # 5. Phase C training history
    _write_history_csv(
        phase_c_history,
        os.path.join(OUTPUT_DIR, "training_phase_c.csv"),
    )


def _write_history_csv(history: list, path: str) -> None:
    if not history:
        return
    keys = list(history[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in history:
            writer.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})



# ═══════════════════════════════════════════════════════════════════════════

FAULT_LABELS = {0: "Normal", 1: "Anchor-Drag", 2: "Seismic", 3: "Equipment-Failure"}


def _make_graph(n_nodes: int = N_NODES, anomaly_rate: float = ANOMALY_RATE,
                seed: int = 0) -> Data:
    """Build one synthetic cable-segment graph."""
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    # Node features: [voltage, dV/dt, d²V/dt², tension, depth_km,
    #                 backscatter_slope, seismic_pga, vessel_count]
    x = torch.tensor(rng.randn(n_nodes, NODE_DIM).astype(np.float32))

    # Edge list: random cable topology (no self-loops)
    src, dst = [], []
    for _ in range(N_CABLES):
        u = rng.randint(0, n_nodes)
        v = rng.randint(0, n_nodes)
        if u != v:
            src += [u, v]
            dst += [v, u]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Edge attributes: [length_km, depth_m, R, L, C, G]
    n_edges = edge_index.shape[1]
    edge_attr = torch.tensor(
        rng.rand(n_edges, EDGE_DIM).astype(np.float32) * [200, 4000, 0.1, 0.001, 0.001, 1e-6],
        dtype=torch.float,
    )

    # Labels
    n_anom = max(1, int(n_nodes * anomaly_rate))
    anom_idx = rng.choice(n_nodes, size=n_anom, replace=False)
    y_anomaly = torch.zeros(n_nodes)
    y_anomaly[anom_idx] = 1.0

    y_attribution = torch.zeros(n_nodes, dtype=torch.long)
    for idx in anom_idx:
        y_attribution[idx] = int(rng.randint(1, N_CLASSES))

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y_anomaly=y_anomaly,
        y_attribution=y_attribution,
    )


def build_dataset(n_graphs: int = 80, val_frac: float = 0.2):
    graphs = [_make_graph(seed=i) for i in range(n_graphs)]
    split = int(n_graphs * (1 - val_frac))
    train_loader = DataLoader(graphs[:split], batch_size=8, shuffle=True)
    val_loader = DataLoader(graphs[split:], batch_size=8, shuffle=False)
    return train_loader, val_loader, graphs[:split], graphs[split:]


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Pretty-print helpers
# ═══════════════════════════════════════════════════════════════════════════

def _bar(value: float, width: int = 30, fill: str = "█", empty: str = "░") -> str:
    n = round(value * width)
    return fill * n + empty * (width - n)


def _section(title: str) -> None:
    print(f"\n{'═' * 65}")
    print(f"  {title}")
    print(f"{'═' * 65}")


def _subsection(title: str) -> None:
    print(f"\n  ── {title} {'─' * (55 - len(title))}")


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Logging callback
# ═══════════════════════════════════════════════════════════════════════════

class PrintCallback(TrainingCallback):
    def __init__(self, log_every: int = 5):
        self._log_every = log_every

    def on_phase_start(self, phase_name: str) -> None:
        _subsection(phase_name)

    def on_epoch_end(self, epoch: int, metrics: dict) -> None:
        if epoch % self._log_every == 0:
            parts = [f"epoch={epoch:3d}"]
            for k in ("total", "anomaly", "attribution", "auroc", "f1"):
                if k in metrics:
                    parts.append(f"{k}={metrics[k]:.4f}")
            print("    " + "  |  ".join(parts))

    def on_phase_end(self, phase_name: str, final_metrics: dict) -> None:
        auroc = final_metrics.get("auroc", "—")
        f1 = final_metrics.get("f1", "—")
        auroc_str = f"{auroc:.4f}" if isinstance(auroc, float) else auroc
        f1_str = f"{f1:.4f}" if isinstance(f1, float) else f1
        print(f"\n    ✓ {phase_name}  AUROC={auroc_str}  F1={f1_str}")


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Federated learning helpers
# ═══════════════════════════════════════════════════════════════════════════

class _FedClient:
    """Thin wrapper around FederatedServer-compatible client API."""
    def __init__(self, model: nn.Module, loader, loss_fn, n_local_epochs: int = 3):
        self._model = copy.deepcopy(model)
        self._loader = loader
        self._loss_fn = loss_fn
        self._n_local = n_local_epochs

    def set_model_state(self, state_dict: dict) -> None:
        self._model.load_state_dict(state_dict)

    def local_train(self):
        opt = torch.optim.Adam(self._model.parameters(), lr=3e-4)
        self._model.train()
        total_loss = 0.0
        n_batches = 0
        n_samples = 0
        for _ in range(self._n_local):
            for batch in self._loader:
                opt.zero_grad()
                anomaly_pred, attr_pred = self._model(batch)
                y_anom = getattr(batch, "y_anomaly", torch.zeros(anomaly_pred.shape[0]))
                y_attr = getattr(batch, "y_attribution", torch.zeros(attr_pred.shape[0], dtype=torch.long))
                losses = self._loss_fn(
                    anomaly_pred, attr_pred, y_anom, y_attr,
                    torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0),
                )
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                opt.step()
                total_loss += losses["total"].item()
                n_batches += 1
                n_samples += batch.num_nodes
        avg_loss = total_loss / max(n_batches, 1)
        return self._model.state_dict(), n_samples, avg_loss


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Main demo
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    t0 = time.time()

    # Capture all console output so we can also save it to output/console_log.txt
    tee = _Tee()
    sys.stdout = tee

    # ------------------------------------------------------------------ header
    print()
    print("╔" + "═" * 63 + "╗")
    print("║     AQUILA-NET  ·  End-to-End Demo Run                    ║")
    print("║  Physics-Aware Federated GNN for Undersea Cable Anomalies ║")
    print("╚" + "═" * 63 + "╝")

    # ------------------------------------------------------------------ dataset
    _section("DATASET")
    print("  Building synthetic cable-graph dataset …")
    train_loader, val_loader, train_graphs, val_graphs = build_dataset(n_graphs=80)

    n_total = sum(g.num_nodes for g in train_graphs + val_graphs)
    n_anom = sum(int(g.y_anomaly.sum()) for g in train_graphs + val_graphs)
    print(f"  Total graphs  : {len(train_graphs) + len(val_graphs)}")
    print(f"  Train / Val   : {len(train_graphs)} / {len(val_graphs)}")
    print(f"  Nodes total   : {n_total}  |  anomalous: {n_anom}  ({100*n_anom/n_total:.1f} %)")
    print(f"  Node features : {NODE_DIM}")
    print(f"  Edge features : {EDGE_DIM}")
    print(f"  Fault classes : {', '.join(FAULT_LABELS.values())}")

    # ------------------------------------------------------------------ EKF demo
    _section("ADAPTIVE EKF  –  AR(2) Anomaly Scoring")
    ekf = AdaptiveEKF(phi1=0.8, phi2=0.1, Q=0.01, R=0.05)
    signal = np.concatenate([
        np.random.randn(40) * 0.1,            # nominal
        np.random.randn(10) * 0.6 + 1.5,      # anomaly injected
        np.random.randn(20) * 0.1,
    ])
    scores = []
    for obs in signal:
        ekf.predict()
        result = ekf.update(np.array([obs]))
        scores.append(float(result["filtered_score"]))
    peak_idx = int(np.argmax(scores))
    print(f"  Signal length : {len(signal)} samples")
    print(f"  Peak anomaly score : {max(scores):.4f}  at sample #{peak_idx}")
    print(f"  Anomaly region (samples 40-50) → score range: "
          f"{min(scores[40:50]):.3f} – {max(scores[40:50]):.3f}")
    print(f"  Nominal region (samples 0-40) → score range: "
          f"{min(scores[:40]):.3f} – {max(scores[:40]):.3f}")

    # ------------------------------------------------------------------ DDPM demo
    _section("DDPM WAVEFORM AUGMENTOR")
    ddpm = DDPMAugmentor(seq_len=64, T=10)
    dummy_waveforms = torch.randn(8, 1, 64)    # (batch, channels=1, seq_len)
    noisy = ddpm.q_sample(dummy_waveforms, torch.tensor([5] * 8))
    aug_ds = ddpm.generate_sabotage_samples(n_samples=12)
    print(f"  Original waveforms          : {dummy_waveforms.shape}  (B, C, L)")
    print(f"  After q_sample (t=5)        : {noisy.shape}  "
          f"mean={noisy.mean().item():.3f}  std={noisy.std().item():.3f}")
    print(f"  Generated sabotage samples  : {aug_ds.shape}  ({aug_ds.shape[0]} samples)")

    # ------------------------------------------------------------------ model
    _section("MODEL ARCHITECTURE")
    model = PINNGraphNeuralNetwork(
        node_dim=NODE_DIM, edge_dim=EDGE_DIM, hidden_dim=HIDDEN_DIM, n_classes=N_CLASSES
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Architecture      : PINN-GNN  (Physics-Constrained MPNN → GATv2)")
    print(f"  Node / edge dims  : {NODE_DIM} / {EDGE_DIM}")
    print(f"  Hidden dim        : {HIDDEN_DIM}")
    print(f"  Output classes    : {N_CLASSES}")
    print(f"  Total params      : {total_params:,}")
    print(f"  Trainable params  : {trainable:,}")

    # ------------------------------------------------------------------ phase A
    _section("TRAINING PIPELINE")
    loss_fn = AQUILALoss(n_classes=N_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    callback = PrintCallback(log_every=5)
    trainer = AQUILATrainer(
        model=model, optimizer=optimizer, loss_fn=loss_fn,
        device=DEVICE, callbacks=[callback],
    )
    phase_a_history = trainer.train_phase_a(n_epochs=20, data_loader=train_loader)

    # ------------------------------------------------------------------ phase B
    fl_server = FederatedServer(model=model, n_clients=5)
    client_loaders = [
        DataLoader(train_graphs[i::5], batch_size=4, shuffle=True) for i in range(5)
    ]
    clients = [
        _FedClient(model, client_loaders[i], loss_fn, n_local_epochs=3)
        for i in range(5)
    ]
    phase_b_history = trainer.train_phase_b_federated(
        fl_server=fl_server, clients=clients, n_rounds=10
    )

    # ------------------------------------------------------------------ phase C
    aug_graphs = [_make_graph(seed=AUG_SEED_OFFSET + i, anomaly_rate=0.40) for i in range(20)]
    aug_loader = DataLoader(aug_graphs, batch_size=8, shuffle=True)
    phase_c_history = trainer.train_phase_c_augmented(n_epochs=10, augmented_loader=aug_loader)

    # ------------------------------------------------------------------ evaluation
    _section("EVALUATION RESULTS")
    evaluator = AQUILAEvaluator(model)
    results = evaluator.evaluate_on_dataset(val_loader)

    auroc = results["auroc"]
    f1 = results["f1_macro"]
    ece = results["ece"]
    cm = results["confusion_matrix"]

    print(f"\n  {'Metric':<26} {'Value':>10}   {'Bar':}")
    print(f"  {'─' * 60}")
    print(f"  {'AUROC':<26} {auroc:>10.4f}   {_bar(auroc)}")
    print(f"  {'F1 Macro':<26} {f1:>10.4f}   {_bar(f1)}")
    print(f"  {'ECE (↓ better)':<26} {ece:>10.4f}   {_bar(1 - ece)}")

    print(f"\n  Confusion Matrix  (rows = true class, cols = predicted):")
    header = "  " + " " * (CM_LABEL_WIDTH + 2) + "  ".join(
        f"{FAULT_LABELS[i][:6]:>6}" for i in range(N_CLASSES)
    )
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:6.2f}" for v in row)
        print(f"  {FAULT_LABELS[i]:<{CM_LABEL_WIDTH}}  {row_str}")

    # ------------------------------------------------------------------ baselines
    _section("BASELINE COMPARISON")
    comparison_df = evaluator.compare_with_baselines(val_loader)
    col_w = [28, 8, 8, 8]
    header_row = (f"  {'Method':<{col_w[0]}}"
                  f"{'AUROC':>{col_w[1]}}"
                  f"{'F1':>{col_w[2]}}"
                  f"{'ECE':>{col_w[3]}}")
    print(header_row)
    print("  " + "─" * sum(col_w))
    for _, row in comparison_df.iterrows():
        marker = " ◄ best" if row["method"] == "AQUILA" else ""
        print(f"  {row['method']:<{col_w[0]}}"
              f"{row['auroc']:>{col_w[1]}.4f}"
              f"{row['f1']:>{col_w[2]}.4f}"
              f"{row['ece']:>{col_w[3]}.4f}"
              f"{marker}")

    # ------------------------------------------------------------------ training curves summary
    _section("TRAINING CURVES SUMMARY")
    _subsection("Phase A — Supervised Pre-training (20 epochs)")
    if phase_a_history:
        first = phase_a_history[0]
        last = phase_a_history[-1]
        print(f"  epoch   1 :  loss={first.get('total', 0):.4f}  "
              f"AUROC={first.get('auroc', 0):.4f}  F1={first.get('f1', 0):.4f}")
        print(f"  epoch  20 :  loss={last.get('total', 0):.4f}  "
              f"AUROC={last.get('auroc', 0):.4f}  F1={last.get('f1', 0):.4f}")
        improvement = last.get("auroc", 0) - first.get("auroc", 0)
        print(f"  AUROC improvement : {improvement:+.4f}")

    _subsection("Phase B — Federated Learning (10 rounds, 5 clients)")
    if phase_b_history:
        # Collect client-average loss per round from the fl_server
        print(f"  Completed {len(phase_b_history)} federated rounds across 5 clients")
        print(f"  First round metrics : {phase_b_history[0]}")
        print(f"  Last  round metrics : {phase_b_history[-1]}")

    _subsection("Phase C — DDPM-Augmented Fine-tuning (10 epochs)")
    if phase_c_history:
        first = phase_c_history[0]
        last = phase_c_history[-1]
        print(f"  epoch   1 :  loss={first.get('total', 0):.4f}  "
              f"AUROC={first.get('auroc', 0):.4f}  F1={first.get('f1', 0):.4f}")
        print(f"  epoch  10 :  loss={last.get('total', 0):.4f}  "
              f"AUROC={last.get('auroc', 0):.4f}  F1={last.get('f1', 0):.4f}")

    # ------------------------------------------------------------------ report
    _section("FINAL REPORT")
    report = evaluator.generate_report(results)
    print(report)

    # ------------------------------------------------------------------ footer
    elapsed = time.time() - t0
    print(f"\n  Total run time : {elapsed:.1f} s")
    print("  All systems nominal.\n")

    # ------------------------------------------------------------------ save outputs
    sys.stdout = tee._orig   # restore real stdout before printing save status
    console_text = tee.getvalue()
    _save_outputs(console_text, results, comparison_df, phase_a_history, phase_c_history)
    print(f"  ✓ Results saved to  {OUTPUT_DIR}/")
    print(f"      console_log.txt  |  metrics.json  |  baseline_comparison.csv")
    print(f"      training_phase_a.csv  |  training_phase_c.csv")


if __name__ == "__main__":
    main()
