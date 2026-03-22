from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch


class CheckpointManager:
    """Saves and loads model/optimizer state dicts to disk."""

    def __init__(self, checkpoint_dir: str = "checkpoints") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict,
        filename: Optional[str] = None,
    ) -> Path:
        """Persist model and optimizer state to a checkpoint file.

        Args:
            model: The model whose ``state_dict`` will be saved.
            optimizer: The optimizer whose ``state_dict`` will be saved.
            epoch: Current training epoch (stored in the checkpoint).
            metrics: Dict of scalar metric values (e.g. ``{'auroc': 0.95}``).
            filename: Override the auto-generated filename.  When ``None``
                      the checkpoint is named ``checkpoint_epoch_{epoch:04d}.pt``.

        Returns:
            :class:`~pathlib.Path` of the written checkpoint file.
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch:04d}.pt"
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
            },
            path,
        )
        return path

    def load(self, path: str) -> dict:
        """Load a checkpoint from *path*.

        Args:
            path: File path to the checkpoint (absolute or relative).

        Returns:
            Dict with keys 'model_state', 'optimizer_state', 'epoch',
            'metrics'.
        """
        return torch.load(str(path), map_location="cpu", weights_only=True)

    def get_best_checkpoint(self, metric: str = "auroc") -> str:
        """Find the checkpoint with the highest value of *metric*.

        Scans all ``*.pt`` files in *checkpoint_dir* and returns the path
        to the one whose stored metrics contain the best (highest) value for
        *metric*.

        Args:
            metric: Key in the checkpoint's ``metrics`` dict to maximise.

        Returns:
            Absolute path string to the best checkpoint file.

        Raises:
            FileNotFoundError: If no checkpoints exist in *checkpoint_dir*.
            KeyError: If none of the checkpoints contain *metric*.
        """
        pt_files = sorted(self.checkpoint_dir.glob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(
                f"No checkpoint files found in '{self.checkpoint_dir}'."
            )

        best_path: Optional[Path] = None
        best_value = float("-inf")

        for pt_file in pt_files:
            try:
                ckpt = torch.load(str(pt_file), map_location="cpu", weights_only=True)
                value = ckpt.get("metrics", {}).get(metric)
                if value is not None and float(value) > best_value:
                    best_value = float(value)
                    best_path = pt_file
            except Exception:
                # Skip corrupt / unreadable checkpoints
                continue

        if best_path is None:
            raise KeyError(
                f"Metric '{metric}' not found in any checkpoint under "
                f"'{self.checkpoint_dir}'."
            )

        return str(best_path)
