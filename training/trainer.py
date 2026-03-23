from __future__ import annotations

from typing import List, Optional

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

from .callbacks import TrainingCallback


class AQUILATrainer:
    """Orchestrates the three-phase AQUILA-NET training pipeline."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn,
        device: str = "cpu",
        callbacks: Optional[List[TrainingCallback]] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = torch.device(device)
        self.callbacks: List[TrainingCallback] = callbacks or []
        self.model.to(self.device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fire(self, event: str, *args, **kwargs) -> None:
        for cb in self.callbacks:
            getattr(cb, event)(*args, **kwargs)

    @staticmethod
    def _to_device(data, device: torch.device):
        """Move a torch_geometric Data object to *device*."""
        if hasattr(data, "to"):
            return data.to(device)
        return data

    # ------------------------------------------------------------------
    # Core training / evaluation
    # ------------------------------------------------------------------

    def train_epoch(self, data_loader, pde_data=None) -> dict:
        """Run one full training epoch and return mean losses.

        For every batch:
          1. Forward pass through model.
          2. Compute supervised + PDE losses.
          3. Back-propagate and step optimizer.

        Args:
            data_loader: Iterable of torch_geometric Data batches.
                         Each batch is expected to carry ``y_anomaly``
                         and ``y_attribution`` label attributes.
            pde_data: Optional pre-computed PDE residual tensors dict with
                      keys 'telegrapher', 'euler_bernoulli', 'causality'.
                      When None all PDE terms are set to zero.

        Returns:
            Dict of mean loss values over the epoch.
        """
        self.model.train()
        accum: dict = {
            "total": 0.0,
            "anomaly": 0.0,
            "attribution": 0.0,
            "telegrapher": 0.0,
            "euler_bernoulli": 0.0,
            "causality": 0.0,
        }
        n_batches = 0

        for batch in data_loader:
            batch = self._to_device(batch, self.device)
            self.optimizer.zero_grad()

            anomaly_pred, attr_pred = self.model(batch)

            anomaly_labels = batch.y_anomaly if hasattr(batch, "y_anomaly") else torch.zeros(anomaly_pred.shape[0], device=self.device)
            attr_labels = batch.y_attribution if hasattr(batch, "y_attribution") else torch.zeros(attr_pred.shape[0], dtype=torch.long, device=self.device)

            if pde_data is not None:
                tele = pde_data.get("telegrapher", torch.tensor(0.0, device=self.device))
                eb = pde_data.get("euler_bernoulli", torch.tensor(0.0, device=self.device))
                caus = pde_data.get("causality", torch.tensor(0.0, device=self.device))
            else:
                zero = torch.tensor(0.0, device=self.device)
                tele, eb, caus = zero, zero, zero

            # Use model's own PDE residuals when available
            if hasattr(self.model, "telegrapher_residual"):
                tele = self.model.telegrapher_residual(batch)
            if hasattr(self.model, "euler_bernoulli_residual"):
                eb = self.model.euler_bernoulli_residual(batch)

            losses = self.loss_fn(
                anomaly_pred, attr_pred,
                anomaly_labels, attr_labels,
                tele, eb, caus,
            )

            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            for k, v in losses.items():
                accum[k] += v.item() if isinstance(v, torch.Tensor) else v
            n_batches += 1

        if n_batches == 0:
            return accum
        return {k: v / n_batches for k, v in accum.items()}

    def evaluate(self, data_loader) -> dict:
        """Evaluate model; returns AUROC and macro-F1.

        Args:
            data_loader: Iterable of torch_geometric Data batches.

        Returns:
            Dict with keys 'auroc' and 'f1'.
        """
        self.model.eval()
        all_anomaly_scores: list = []
        all_anomaly_labels: list = []
        all_attr_preds: list = []
        all_attr_labels: list = []

        with torch.no_grad():
            for batch in data_loader:
                batch = self._to_device(batch, self.device)
                anomaly_pred, attr_pred = self.model(batch)

                scores = torch.sigmoid(anomaly_pred).squeeze().cpu().numpy()
                preds = attr_pred.argmax(dim=-1).cpu().numpy()

                anomaly_labels = (
                    batch.y_anomaly.cpu().numpy()
                    if hasattr(batch, "y_anomaly")
                    else np.zeros(len(scores))
                )
                attr_labels = (
                    batch.y_attribution.cpu().numpy()
                    if hasattr(batch, "y_attribution")
                    else np.zeros(len(preds), dtype=int)
                )

                all_anomaly_scores.append(np.atleast_1d(scores))
                all_anomaly_labels.append(np.atleast_1d(anomaly_labels))
                all_attr_preds.append(np.atleast_1d(preds))
                all_attr_labels.append(np.atleast_1d(attr_labels))

        y_score = np.concatenate(all_anomaly_scores)
        y_true = np.concatenate(all_anomaly_labels)
        y_pred = np.concatenate(all_attr_preds)
        y_attr = np.concatenate(all_attr_labels)

        # Guard against single-class edge cases
        n_pos = int(y_true.sum())
        if n_pos == 0 or n_pos == len(y_true):
            auroc = 0.5
        else:
            auroc = float(roc_auc_score(y_true, y_score))

        f1 = float(f1_score(y_attr, y_pred, average="macro", zero_division=0))
        return {"auroc": auroc, "f1": f1}

    # ------------------------------------------------------------------
    # Three-phase training API
    # ------------------------------------------------------------------

    def train_phase_a(
        self,
        n_epochs: int = 50,
        data_loader=None,
    ) -> List[dict]:
        """Phase A: supervised pre-training on labelled fault data.

        Args:
            n_epochs: Number of epochs to run.
            data_loader: DataLoader supplying labelled batches.

        Returns:
            List of per-epoch metric dicts.
        """
        self._fire("on_phase_start", "Phase A – Supervised Pre-training")
        history: List[dict] = []

        for epoch in range(1, n_epochs + 1):
            train_metrics = self.train_epoch(data_loader)
            eval_metrics = self.evaluate(data_loader) if data_loader is not None else {}
            combined = {**train_metrics, **eval_metrics}
            history.append({"epoch": epoch, **combined})
            self._fire("on_epoch_end", epoch, combined)

        final = history[-1] if history else {}
        self._fire("on_phase_end", "Phase A – Supervised Pre-training", final)
        return history

    def train_phase_b_federated(
        self,
        fl_server,
        clients: list,
        n_rounds: int = 100,
    ) -> List[dict]:
        """Phase B: federated learning across submarine-cable operators.

        Each round:
          1. Distribute global weights to selected clients.
          2. Each client trains locally and returns an update.
          3. Server aggregates updates (FedAvg).
          4. Evaluation on server-side held-out data (if available).

        Args:
            fl_server: FederatedServer instance.
            clients: List of FLClient instances.
            n_rounds: Number of federated rounds.

        Returns:
            List of per-round metric dicts.
        """
        self._fire("on_phase_start", "Phase B – Federated Learning")
        history: List[dict] = []

        for rnd in range(1, n_rounds + 1):
            global_state = fl_server.model.state_dict()

            client_updates = []
            for client in clients:
                if hasattr(client, "set_model_state"):
                    client.set_model_state(global_state)
                update = client.local_train()
                if isinstance(update, tuple):
                    client_updates.append(update)
                else:
                    client_updates.append((update, 1, 0.0))

            if client_updates:
                new_state = fl_server.aggregate_weights(client_updates)
                fl_server.model.load_state_dict(new_state)
                self.model.load_state_dict(new_state)

            round_metrics: dict = {"round": rnd}
            if hasattr(fl_server, "round_metrics") and fl_server.round_metrics:
                round_metrics.update(fl_server.round_metrics[-1])

            history.append(round_metrics)
            self._fire("on_epoch_end", rnd, {k: v for k, v in round_metrics.items() if isinstance(v, (int, float))})

        final = history[-1] if history else {}
        self._fire("on_phase_end", "Phase B – Federated Learning", {k: v for k, v in final.items() if isinstance(v, (int, float))})
        return history

    def train_phase_c_augmented(
        self,
        n_epochs: int = 30,
        augmented_loader=None,
    ) -> List[dict]:
        """Phase C: fine-tuning on DDPM-augmented sabotage samples.

        Training focuses on improving sabotage recall by training on
        augmented data produced by the cable-conditioned diffusion model.

        Args:
            n_epochs: Fine-tuning epochs.
            augmented_loader: DataLoader with augmented (synthetic) batches.

        Returns:
            List of per-epoch metric dicts.
        """
        self._fire("on_phase_start", "Phase C – Augmented Fine-tuning (Sabotage Recall)")
        history: List[dict] = []

        for epoch in range(1, n_epochs + 1):
            train_metrics = self.train_epoch(augmented_loader)
            eval_metrics = self.evaluate(augmented_loader) if augmented_loader is not None else {}
            combined = {**train_metrics, **eval_metrics}
            history.append({"epoch": epoch, **combined})
            self._fire("on_epoch_end", epoch, combined)

        final = history[-1] if history else {}
        self._fire("on_phase_end", "Phase C – Augmented Fine-tuning (Sabotage Recall)", final)
        return history
