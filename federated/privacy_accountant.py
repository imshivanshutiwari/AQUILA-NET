from typing import Dict


class PrivacyAccountant:
    """Tracks cumulative privacy budget (epsilon, delta) across federated learning rounds."""

    def __init__(self, delta: float = 1e-5, target_epsilon: float = 1.0):
        self.delta = delta
        self.target_epsilon = target_epsilon
        self._total_epsilon: float = 0.0
        self._rounds: int = 0

    def add_round(self, noise_multiplier: float, sample_rate: float, n_steps: int) -> None:
        """Accumulate privacy cost for one training round using an RDP-style approximation.

        Approximate epsilon per step: epsilon = sample_rate^2 * n_steps / (2 * noise_multiplier^2)
        """
        epsilon_step = (sample_rate ** 2 * n_steps) / (2.0 * noise_multiplier ** 2)
        self._total_epsilon += epsilon_step
        self._rounds += 1

    def get_epsilon(self) -> float:
        """Return total epsilon consumed so far."""
        return self._total_epsilon

    def get_remaining_budget(self) -> float:
        """Return remaining epsilon budget (clamped to zero)."""
        return max(0.0, self.target_epsilon - self._total_epsilon)

    def is_budget_exhausted(self) -> bool:
        """Return True when accumulated epsilon meets or exceeds the target."""
        return self._total_epsilon >= self.target_epsilon

    def get_privacy_report(self) -> Dict:
        """Return a summary dict with epsilon, delta, rounds, and remaining budget."""
        return {
            "epsilon": self._total_epsilon,
            "delta": self.delta,
            "rounds": self._rounds,
            "remaining": self.get_remaining_budget(),
        }
