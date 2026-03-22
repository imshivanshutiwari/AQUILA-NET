from typing import Dict, List

import numpy as np

from filters.ar2_model import AR2Model


class AdaptiveEKF:
    """Adaptive Extended Kalman Filter with CUSUM change detection.

    State vector x = [anomaly_score, d_score/dt, trend_component].
    """

    def __init__(
        self,
        phi1: float = 0.8,
        phi2: float = 0.1,
        Q: float = 0.01,
        R: float = 0.05,
        fading_lambda: float = 0.95,
        threshold: float = 0.7,
        cusum_h: float = 5.0,
    ):
        self.ar2 = AR2Model(phi1, phi2)
        self.Q_base = Q * np.eye(3)
        self.R = R
        self.fading_lambda = fading_lambda
        self.threshold = threshold
        self.cusum_h = cusum_h

        self.x = np.zeros(3)
        self.P = np.eye(3) * 0.1
        self.cusum_S: float = 0.0
        self.cusum_mu: float = 0.0
        self.cusum_k: float = 0.5
        self.history: List[Dict] = []

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _build_transition_matrix(self) -> np.ndarray:
        """Build the 3×3 state-transition matrix F."""
        F = np.eye(3)
        F[0, 0] = self.ar2.phi1   # AR coefficient for score
        F[0, 1] = self.ar2.phi2   # AR coefficient from velocity
        F[1, 0] = 1.0             # velocity approximation
        F[2, 2] = 0.99            # trend decay
        return F

    # ── Kalman predict / update ──────────────────────────────────────────────

    def predict(self):
        """Time-update (predict) step.

        Returns:
            (x_pred, P_pred)
        """
        F = self._build_transition_matrix()
        x_pred = F @ self.x
        Q = self.Q_base / self.fading_lambda
        P_pred = F @ self.P @ F.T + Q
        return x_pred, P_pred

    def update(self, measurement: float) -> Dict:
        """Measurement-update (correct) step followed by CUSUM monitoring.

        Args:
            measurement: scalar observation of the anomaly score.

        Returns:
            dict with keys: filtered_score, covariance, cusum, change_detected,
            anomaly_detected.
        """
        x_pred, P_pred = self.predict()

        H = np.array([[1.0, 0.0, 0.0]])           # observation matrix (1×3)
        innovation = measurement - (H @ x_pred).item()
        S_scalar = (H @ P_pred @ H.T).item() + self.R
        K = (P_pred @ H.T) / S_scalar             # Kalman gain (3×1)

        self.x = x_pred + K.flatten() * innovation
        self.P = (np.eye(3) - np.outer(K.flatten(), H)) @ P_pred

        # ── CUSUM ────────────────────────────────────────────────────────────
        self.cusum_S = max(
            0.0,
            self.cusum_S + self.x[0] - self.cusum_mu - self.cusum_k,
        )
        change_detected = self.cusum_S > self.cusum_h

        record = {
            "score": float(self.x[0]),
            "cusum": float(self.cusum_S),
            "change_detected": bool(change_detected),
        }
        self.history.append(record)

        return {
            "filtered_score": float(self.x[0]),
            "covariance": float(self.P[0, 0]),
            "cusum": float(self.cusum_S),
            "change_detected": bool(change_detected),
            "anomaly_detected": bool(self.x[0] > self.threshold),
        }

    # ── Utilities ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset state, covariance, CUSUM accumulator, and history."""
        self.x = np.zeros(3)
        self.P = np.eye(3) * 0.1
        self.cusum_S = 0.0
        self.history = []

    def compute_mttd(self, true_event_time: int, detect_times: List[int]) -> float:
        """Mean time-to-detection from *true_event_time*.

        Finds the first detection timestep at or after *true_event_time* and
        returns the delay.  Returns inf when no detection occurs after the event.
        """
        detect_after = [t for t in detect_times if t >= true_event_time]
        if not detect_after:
            return float("inf")
        return float(min(detect_after) - true_event_time)
