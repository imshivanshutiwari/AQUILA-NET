from typing import Optional, Tuple

import torch
import torch.nn.functional as F


class FederatedClient:
    """Local participant in a federated learning round."""

    def __init__(
        self,
        client_id: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        local_data,
        config: Optional[dict] = None,
    ):
        self.client_id = client_id
        self.model = model
        self.optimizer = optimizer
        self.local_data = local_data
        self.config = config or {}
        self.round_losses: list = []

    def local_train(self, global_weights: dict, n_epochs: int = 5) -> Tuple[dict, int, float]:
        """Load *global_weights*, train locally, and return updated parameters.

        Returns:
            (state_dict, n_samples, final_loss)
        """
        self.apply_weights(global_weights)
        self.model.train()

        device = next(self.model.parameters()).device
        n_samples = 0
        final_loss = 0.0

        for _ in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for batch in self.local_data:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    inputs, labels = batch[0], batch[1]
                else:
                    # Unsupervised / autoencoder-style: reconstruct the input
                    inputs = batch
                    labels = batch

                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(device)
                if isinstance(labels, torch.Tensor):
                    labels = labels.to(device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                loss = self._compute_loss(outputs, labels)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1
                if hasattr(inputs, "__len__"):
                    n_samples += len(inputs)

            if n_batches > 0:
                final_loss = epoch_loss / n_batches

        self.round_losses.append(final_loss)
        return self.model.state_dict(), n_samples, final_loss

    def _compute_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute BCE + CE combined loss, adapting to the output dimensionality."""
        outputs = outputs.float()

        # Determine if the task is binary or multi-class from the output shape
        out_flat = outputs.view(outputs.shape[0], -1)
        if out_flat.shape[-1] == 1 or (labels.ndim == 1 and labels.unique().numel() <= 2):
            # Binary: BCE
            bce = F.binary_cross_entropy_with_logits(
                out_flat.squeeze(-1), labels.float().view(-1)
            )
            return bce
        else:
            # Multi-class: CE
            ce = F.cross_entropy(out_flat, labels.long().view(-1))
            # Also compute BCE averaged over logits as a regularising term
            bce = F.binary_cross_entropy_with_logits(
                out_flat, torch.zeros_like(out_flat).scatter_(
                    1,
                    labels.long().view(-1, 1).clamp(0, out_flat.shape[-1] - 1),
                    1.0,
                )
            )
            return 0.5 * ce + 0.5 * bce

    def apply_weights(self, weights: dict) -> None:
        """Load a state dict into the local model."""
        self.model.load_state_dict(weights)

    def get_data_size(self) -> int:
        """Return the number of local samples (best-effort)."""
        return len(self.local_data) if hasattr(self.local_data, "__len__") else 0
