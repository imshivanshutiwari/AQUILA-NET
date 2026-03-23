from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F

from .metrics import AQUILAMetrics


class _SimpleGCN(nn.Module):
    """Two-layer GCN baseline without physics constraints."""

    def __init__(self, in_channels: int, hidden: int = 64, n_classes: int = 4):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.anomaly_head = nn.Linear(hidden, 1)
        self.attr_head = nn.Linear(hidden, n_classes)

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        return self.anomaly_head(x), self.attr_head(x)


class AQUILAEvaluator:
    """Evaluates an AQUILA model and compares it against standard baselines."""

    def __init__(self, model: nn.Module, metrics: AQUILAMetrics | None = None):
        self.model = model
        self.metrics = metrics or AQUILAMetrics()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _collect_predictions(self, data_loader, model=None, device="cpu"):
        """Run *model* on *data_loader* and return stacked arrays."""
        m = model or self.model
        m.eval()
        scores, labels, attr_preds, attr_labels = [], [], [], []
        with torch.no_grad():
            for batch in data_loader:
                if hasattr(batch, "to"):
                    batch = batch.to(device)
                anomaly_out, attr_out = m(batch)
                scores.append(torch.sigmoid(anomaly_out).squeeze().cpu().numpy())
                attr_preds.append(attr_out.argmax(-1).cpu().numpy())
                if hasattr(batch, "y_anomaly"):
                    labels.append(batch.y_anomaly.cpu().numpy())
                else:
                    labels.append(np.zeros(len(scores[-1])))
                if hasattr(batch, "y_attribution"):
                    attr_labels.append(batch.y_attribution.cpu().numpy())
                else:
                    attr_labels.append(np.zeros(len(attr_preds[-1]), dtype=int))
        return (
            np.concatenate(scores),
            np.concatenate(labels),
            np.concatenate(attr_preds),
            np.concatenate(attr_labels),
        )

    def _collect_node_features(self, data_loader):
        """Collect flat node features and labels for sklearn baselines."""
        Xs, y_anom, y_attr = [], [], []
        for batch in data_loader:
            x_np = batch.x.cpu().numpy()
            Xs.append(x_np)
            if hasattr(batch, "y_anomaly"):
                y_anom.append(batch.y_anomaly.cpu().numpy())
            else:
                y_anom.append(np.zeros(len(x_np)))
            if hasattr(batch, "y_attribution"):
                y_attr.append(batch.y_attribution.cpu().numpy())
            else:
                y_attr.append(np.zeros(len(x_np), dtype=int))
        return (
            np.concatenate(Xs),
            np.concatenate(y_anom),
            np.concatenate(y_attr).astype(int),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_on_dataset(self, data_loader) -> dict:
        """Run model inference and compute the full metric suite.

        Args:
            data_loader: DataLoader of torch_geometric Data batches.

        Returns:
            Metrics dict from :meth:`AQUILAMetrics.compute_all_metrics`.
        """
        y_score, y_true, y_pred_attr, y_true_attr = self._collect_predictions(
            data_loader
        )
        return self.metrics.compute_all_metrics(
            y_true, y_score, y_true_attr, y_pred_attr, y_score
        )

    def compare_with_baselines(self, data_loader) -> pd.DataFrame:
        """Compare AQUILA against three baseline methods.

        Baselines:
            - ``SVM``: SVC on raw node features.
            - ``Random Forest``: RFC on raw node features.
            - ``Simple GCN``: 2-layer GCN without physics constraints.

        Args:
            data_loader: DataLoader used for both training and evaluation
                         (small datasets, e.g. validation split).

        Returns:
            DataFrame with columns: method, auroc, f1, ece.
        """
        X, y_anom, y_attr = self._collect_node_features(data_loader)
        rows = []

        # --- AQUILA ---
        y_score, y_true, y_pred_attr, y_true_attr = self._collect_predictions(
            data_loader
        )
        rows.append({
            "method": "AQUILA",
            "auroc": self.metrics.compute_auroc(y_true, y_score),
            "f1": self.metrics.compute_f1(y_true_attr, y_pred_attr),
            "ece": self.metrics.compute_ece(y_true, y_score),
        })

        # --- SVM ---
        svm = SVC(kernel="rbf", probability=True, max_iter=1000)
        svm.fit(X, y_anom)
        svm_scores = svm.predict_proba(X)[:, 1]
        svm_preds = svm.predict(X).astype(int)
        rows.append({
            "method": "SVM (OTDR features)",
            "auroc": self.metrics.compute_auroc(y_anom, svm_scores),
            "f1": self.metrics.compute_f1(y_attr, svm_preds % 4),
            "ece": self.metrics.compute_ece(y_anom, svm_scores),
        })

        # --- Random Forest ---
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y_anom)
        rf_scores = rf.predict_proba(X)[:, 1]
        rf_preds = rf.predict(X).astype(int)
        rows.append({
            "method": "Random Forest",
            "auroc": self.metrics.compute_auroc(y_anom, rf_scores),
            "f1": self.metrics.compute_f1(y_attr, rf_preds % 4),
            "ece": self.metrics.compute_ece(y_anom, rf_scores),
        })

        # --- Simple GCN ---
        in_dim = X.shape[1]
        gcn = _SimpleGCN(in_dim)
        gcn_scores, gcn_true, gcn_pred, gcn_attr = self._collect_predictions(
            data_loader, model=gcn
        )
        rows.append({
            "method": "Simple GCN (no physics)",
            "auroc": self.metrics.compute_auroc(gcn_true, gcn_scores),
            "f1": self.metrics.compute_f1(gcn_attr, gcn_pred),
            "ece": self.metrics.compute_ece(gcn_true, gcn_scores),
        })

        return pd.DataFrame(rows, columns=["method", "auroc", "f1", "ece"])

    def generate_report(self, results: dict) -> str:
        """Format an evaluation results dict into a human-readable report.

        Args:
            results: Dict as returned by :meth:`evaluate_on_dataset`.

        Returns:
            Multi-line report string.
        """
        lines = [
            "=" * 60,
            "          AQUILA-NET Evaluation Report",
            "=" * 60,
        ]
        scalar_keys = ["auroc", "f1_macro", "ece"]
        for k in scalar_keys:
            if k in results:
                lines.append(f"  {k:<20s}: {results[k]:.4f}")
        if "confusion_matrix" in results:
            lines.append("\n  Confusion Matrix (normalised by true class):")
            cm = results["confusion_matrix"]
            for row in cm:
                lines.append("    " + "  ".join(f"{v:.2f}" for v in row))
        lines.append("=" * 60)
        return "\n".join(lines)
