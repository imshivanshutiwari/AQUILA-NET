import numpy as np


class AR2Model:
    """AR(2) process: s(t) = phi1*s(t-1) + phi2*s(t-2) + epsilon."""

    def __init__(self, phi1: float = 0.8, phi2: float = 0.1, sigma_eps: float = 0.1):
        self.phi1 = phi1
        self.phi2 = phi2
        self.sigma_eps = sigma_eps

    def is_stationary(self) -> bool:
        """AR(2) is stationary iff |phi2|<1, phi2+phi1<1, phi2-phi1<1."""
        return (
            abs(self.phi2) < 1
            and self.phi2 + self.phi1 < 1
            and self.phi2 - self.phi1 < 1
        )

    def predict_next(self, s_prev: float, s_prev2: float) -> float:
        """One-step-ahead prediction (noise-free)."""
        return self.phi1 * s_prev + self.phi2 * s_prev2

    def compute_transition_matrix(self) -> np.ndarray:
        """Companion / state-transition matrix F."""
        return np.array([[self.phi1, self.phi2], [1.0, 0.0]])

    def compute_noise_covariance(self) -> np.ndarray:
        """Process-noise covariance Q for the companion state."""
        return np.array([[self.sigma_eps ** 2, 0.0], [0.0, 0.0]])
