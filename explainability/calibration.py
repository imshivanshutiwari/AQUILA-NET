from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


class ModelCalibrator:
    """Post-hoc calibration utilities for probabilistic classifiers."""

    def __init__(self, n_bins: int = 10) -> None:
        self.n_bins = n_bins
        # Populated after calibrate / plot_reliability_diagram calls
        self._bin_centers: np.ndarray | None = None
        self._acc_per_bin: np.ndarray | None = None
        self._conf_per_bin: np.ndarray | None = None

    # ------------------------------------------------------------------
    # ECE
    # ------------------------------------------------------------------

    def compute_ece(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Expected Calibration Error.

        Partitions the probability range [0, 1] into *n_bins* equal-width
        bins and returns the sample-weighted mean absolute gap between
        accuracy and mean confidence within each bin.

        Args:
            y_true: Binary ground-truth labels (0/1).
            y_prob: Predicted probabilities in [0, 1].

        Returns:
            ECE value in [0, 1].
        """
        y_true = np.asarray(y_true, dtype=float)
        y_prob = np.asarray(y_prob, dtype=float)
        n = len(y_true)
        if n == 0:
            return 0.0

        bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        acc_per_bin = np.zeros(self.n_bins)
        conf_per_bin = np.zeros(self.n_bins)
        ece = 0.0

        for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            mask = (y_prob >= lo) & (y_prob < hi)
            if hi == 1.0:
                mask |= y_prob == 1.0
            n_b = mask.sum()
            if n_b == 0:
                continue
            acc_b = y_true[mask].mean()
            conf_b = y_prob[mask].mean()
            acc_per_bin[i] = acc_b
            conf_per_bin[i] = conf_b
            ece += (n_b / n) * abs(acc_b - conf_b)

        self._bin_centers = centers
        self._acc_per_bin = acc_per_bin
        self._conf_per_bin = conf_per_bin
        return float(ece)

    # ------------------------------------------------------------------
    # Temperature scaling
    # ------------------------------------------------------------------

    def temperature_scaling(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Find the optimal temperature for Platt/temperature scaling.

        Minimises the binary cross-entropy of the calibrated probabilities
        ``sigmoid(logits / T)`` with respect to scalar temperature *T* using
        ``scipy.optimize.minimize_scalar`` with the bounded method.

        Args:
            logits: Raw (uncalibrated) logit scores, shape (N,).
            labels: Binary ground-truth labels, shape (N,).

        Returns:
            Optimal temperature T > 0.
        """
        logits = np.asarray(logits, dtype=float).ravel()
        labels = np.asarray(labels, dtype=float).ravel()
        eps = 1e-7

        def nll(T: float) -> float:
            p = 1.0 / (1.0 + np.exp(-logits / T))
            p = np.clip(p, eps, 1.0 - eps)
            return -np.mean(labels * np.log(p) + (1.0 - labels) * np.log(1.0 - p))

        result = minimize_scalar(nll, bounds=(1e-2, 10.0), method="bounded")
        return float(result.x)

    # ------------------------------------------------------------------
    # Reliability diagram data
    # ------------------------------------------------------------------

    def plot_reliability_diagram(self) -> dict:
        """Return data needed to render a reliability (calibration) diagram.

        Requires :meth:`compute_ece` to have been called at least once so that
        the internal bin statistics are populated.

        Returns:
            Dict with keys 'bin_centers', 'accuracy_per_bin',
            'confidence_per_bin'.  All values are numpy arrays of length
            *n_bins*.

        Raises:
            RuntimeError: If :meth:`compute_ece` has not been called yet.
        """
        if self._bin_centers is None:
            raise RuntimeError(
                "Call compute_ece() first to populate bin statistics."
            )
        return {
            "bin_centers": self._bin_centers,
            "accuracy_per_bin": self._acc_per_bin,
            "confidence_per_bin": self._conf_per_bin,
        }
