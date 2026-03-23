import torch
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np


class AnomalyDetector:
    """Threshold-based anomaly detector with AUROC and F1 evaluation."""

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def predict(self, anomaly_scores: torch.Tensor) -> torch.Tensor:
        """Classify nodes as anomalous when their score exceeds the threshold.

        Args:
            anomaly_scores: tensor of shape (N,) or (N, 1) with values in [0, 1].

        Returns:
            Boolean tensor of the same batch dimension, True where anomalous.
        """
        scores = anomaly_scores.squeeze(-1)
        return scores > self.threshold

    def compute_auroc(self, scores, labels) -> float:
        """Compute the Area Under the ROC Curve.

        Args:
            scores: tensor or array-like of continuous anomaly scores, shape (N,).
            labels: tensor or array-like of binary ground-truth labels, shape (N,).

        Returns:
            AUROC as a float in [0, 1].
        """
        scores_np = _to_numpy(scores)
        labels_np = _to_numpy(labels)
        return float(roc_auc_score(labels_np, scores_np))

    def compute_f1(self, scores, labels, threshold: float = None) -> float:
        """Compute binary F1 score.

        Args:
            scores: tensor or array-like of continuous anomaly scores.
            labels: tensor or array-like of binary ground-truth labels.
            threshold: decision threshold; defaults to ``self.threshold``.

        Returns:
            F1 score as a float.
        """
        if threshold is None:
            threshold = self.threshold
        scores_np = _to_numpy(scores)
        labels_np = _to_numpy(labels)
        preds = (scores_np >= threshold).astype(int)
        return float(f1_score(labels_np, preds, zero_division=0))


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().ravel()
    return np.asarray(x).ravel()
