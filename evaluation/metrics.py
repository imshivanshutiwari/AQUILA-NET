from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_recall_curve,
    confusion_matrix,
)


class AQUILAMetrics:
    """Collection of evaluation metrics for anomaly detection and attribution."""

    @staticmethod
    def compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Area under the ROC curve.

        Args:
            y_true: Binary ground-truth labels.
            y_score: Predicted probability scores.

        Returns:
            AUROC in [0, 1]; returns 0.5 when only one class is present.
        """
        n_pos = int(np.sum(y_true))
        if n_pos == 0 or n_pos == len(y_true):
            return 0.5
        return float(roc_auc_score(y_true, y_score))

    @staticmethod
    def compute_f1(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = "macro",
    ) -> float:
        """Macro (or other) F1 score.

        Args:
            y_true: Integer class labels.
            y_pred: Predicted class labels.
            average: Averaging strategy passed to sklearn.

        Returns:
            F1 score.
        """
        return float(f1_score(y_true, y_pred, average=average, zero_division=0))

    @staticmethod
    def compute_precision_recall(
        y_true: np.ndarray,
        y_score: np.ndarray,
    ) -> tuple:
        """Precision-recall curve arrays.

        Returns:
            (precision, recall, thresholds) as numpy arrays.
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        return precision, recall, thresholds

    @staticmethod
    def compute_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: str = "true",
    ) -> np.ndarray:
        """Normalised confusion matrix.

        Args:
            y_true: Ground-truth class labels.
            y_pred: Predicted class labels.
            normalize: Passed to sklearn; 'true' normalises over rows.

        Returns:
            Confusion matrix as a float numpy array.
        """
        return confusion_matrix(y_true, y_pred, normalize=normalize)

    @staticmethod
    def compute_ece(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Expected Calibration Error.

        Partitions predictions into *n_bins* equal-width confidence bins and
        computes the weighted mean absolute difference between accuracy and
        mean confidence within each bin.

        Args:
            y_true: Binary ground-truth labels.
            y_prob: Predicted probabilities in [0, 1].
            n_bins: Number of calibration bins.

        Returns:
            ECE value in [0, 1].
        """
        y_true = np.asarray(y_true, dtype=float)
        y_prob = np.asarray(y_prob, dtype=float)
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n = len(y_true)
        if n == 0:
            return 0.0
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (y_prob >= lo) & (y_prob < hi)
            # include the rightmost edge in the last bin
            if hi == 1.0:
                mask |= y_prob == 1.0
            n_b = mask.sum()
            if n_b == 0:
                continue
            acc_b = y_true[mask].mean()
            conf_b = y_prob[mask].mean()
            ece += (n_b / n) * abs(acc_b - conf_b)
        return float(ece)

    def compute_all_metrics(
        self,
        y_true_anomaly: np.ndarray,
        y_score_anomaly: np.ndarray,
        y_true_attr: np.ndarray,
        y_pred_attr: np.ndarray,
        y_prob_attr: np.ndarray,
    ) -> dict:
        """Compute the full AQUILA metric suite in one call.

        Args:
            y_true_anomaly: Binary anomaly ground truth.
            y_score_anomaly: Anomaly probability scores.
            y_true_attr: Multi-class attribution ground truth.
            y_pred_attr: Predicted attribution class indices.
            y_prob_attr: Predicted attribution probabilities (for ECE).

        Returns:
            Dict with keys: auroc, f1_macro, ece, precision, recall,
            thresholds, confusion_matrix.
        """
        auroc = self.compute_auroc(y_true_anomaly, y_score_anomaly)
        f1 = self.compute_f1(y_true_attr, y_pred_attr, average="macro")
        ece = self.compute_ece(y_true_anomaly, y_score_anomaly)
        precision, recall, thresholds = self.compute_precision_recall(
            y_true_anomaly, y_score_anomaly
        )
        cm = self.compute_confusion_matrix(y_true_attr, y_pred_attr)
        return {
            "auroc": auroc,
            "f1_macro": f1,
            "ece": ece,
            "precision": precision,
            "recall": recall,
            "thresholds": thresholds,
            "confusion_matrix": cm,
        }
