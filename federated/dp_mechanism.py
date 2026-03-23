import torch


class DifferentialPrivacyMechanism:
    """Wraps Opacus PrivacyEngine to add DP-SGD training to a PyTorch model."""

    def __init__(
        self,
        noise_multiplier: float = 1.1,
        max_grad_norm: float = 1.0,
        delta: float = 1e-5,
    ):
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.delta = delta
        self._engine = None

    def make_private(self, model, optimizer, dataloader):
        """Attach the Opacus PrivacyEngine and return DP-wrapped objects.

        Returns:
            (priv_model, priv_optimizer, priv_loader)
        """
        from opacus import PrivacyEngine

        engine = PrivacyEngine()
        priv_model, priv_optimizer, priv_loader = engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
        )
        self._engine = engine
        return priv_model, priv_optimizer, priv_loader

    def get_epsilon(self) -> float:
        """Return the privacy cost epsilon spent so far."""
        if self._engine is None:
            raise RuntimeError("make_private() must be called before get_epsilon().")
        return self._engine.get_epsilon(delta=self.delta)

    def is_budget_safe(self, epsilon_target: float = 1.0) -> bool:
        """Return True when the consumed epsilon is within the target budget."""
        return self.get_epsilon() <= epsilon_target

    def clip_gradients(self, model: torch.nn.Module) -> None:
        """Manually clip per-parameter gradients to max_grad_norm."""
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
