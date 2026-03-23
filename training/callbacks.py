from __future__ import annotations

import pandas as pd


class TrainingCallback:
    """Base training callback with default print-based implementations."""

    def on_epoch_end(self, epoch: int, metrics: dict) -> None:
        """Print epoch number and all metric values."""
        metrics_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"[Epoch {epoch:04d}] {metrics_str}")

    def on_phase_start(self, phase_name: str) -> None:
        """Print a header when a training phase begins."""
        print(f"\n{'='*60}")
        print(f"  Starting phase: {phase_name}")
        print(f"{'='*60}")

    def on_phase_end(self, phase_name: str, metrics: dict) -> None:
        """Print a summary when a training phase ends."""
        print(f"\n--- Phase '{phase_name}' complete ---")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print()


class DashboardCallback(TrainingCallback):
    """Callback that stores per-epoch metrics for dashboard display."""

    def __init__(self) -> None:
        self.epoch_metrics: list[dict] = []

    def on_epoch_end(self, epoch: int, metrics: dict) -> None:
        """Append epoch metrics to internal store and print summary."""
        self.epoch_metrics.append({"epoch": epoch, **metrics})
        super().on_epoch_end(epoch, metrics)

    def get_metrics_df(self) -> pd.DataFrame:
        """Return collected metrics as a pandas DataFrame."""
        return pd.DataFrame(self.epoch_metrics)
