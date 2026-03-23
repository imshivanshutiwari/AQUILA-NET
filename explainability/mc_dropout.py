from __future__ import annotations

import numpy as np
import torch


class MCDropoutUncertainty:
    """Bayesian uncertainty estimation via MC-Dropout inference.

    Performs *T* stochastic forward passes with dropout enabled at test time
    to approximate a posterior predictive distribution.
    """

    def __init__(self, model: torch.nn.Module, T: int = 50) -> None:
        self.model = model
        self.T = T

    def enable_dropout(self) -> None:
        """Switch all Dropout layers to training mode while keeping BatchNorm etc. frozen."""
        for m in self.model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()

    def predict_with_uncertainty(self, data) -> dict:
        """Run *T* stochastic forward passes and return uncertainty estimates.

        Epistemic uncertainty is estimated as the variance of the stochastic
        anomaly predictions across the *T* passes.  Aleatoric uncertainty is
        estimated as the expected Bernoulli variance ``p*(1-p)``.

        Args:
            data: torch_geometric Data object passed directly to the model.

        Returns:
            Dict with keys:
                'mean'             – mean anomaly probability, shape (N, 1).
                'epistemic'        – variance of anomaly probability, shape (N, 1).
                'aleatoric'        – expected Bernoulli variance, shape (N, 1).
                'attribution_mean' – mean attribution logits, shape (N, C).
                'attribution_std'  – std of attribution logits, shape (N, C).
        """
        self.model.eval()
        self.enable_dropout()

        anomaly_preds: list[np.ndarray] = []
        attr_preds: list[np.ndarray] = []

        with torch.no_grad():
            for _ in range(self.T):
                a, b = self.model(data)
                anomaly_preds.append(a.cpu().numpy())
                attr_preds.append(b.cpu().numpy())

        anomaly_arr = np.stack(anomaly_preds)   # (T, N, 1)
        attr_arr = np.stack(attr_preds)          # (T, N, C)

        mean_anomaly = anomaly_arr.mean(axis=0)
        epistemic = anomaly_arr.var(axis=0)
        aleatoric = (anomaly_arr * (1.0 - anomaly_arr)).mean(axis=0)

        return {
            "mean": mean_anomaly,
            "epistemic": epistemic,
            "aleatoric": aleatoric,
            "attribution_mean": attr_arr.mean(axis=0),
            "attribution_std": attr_arr.std(axis=0),
        }
