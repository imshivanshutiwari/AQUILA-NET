import torch
import torch.nn as nn


class AQUILALoss:
    """Combined loss for PINN-GNN training.

    Blends supervised anomaly detection, multi-class attribution, and three
    physics-informed PDE residual terms (Telegrapher, Euler-Bernoulli,
    causality).
    """

    def __init__(
        self,
        lambda_telegrapher: float = 1.0,
        lambda_euler_bernoulli: float = 0.8,
        lambda_causality: float = 0.6,
        anomaly_pos_weight: float = 99.0,
        n_classes: int = 4,
    ):
        self.lambda_telegrapher = lambda_telegrapher
        self.lambda_euler_bernoulli = lambda_euler_bernoulli
        self.lambda_causality = lambda_causality
        self.n_classes = n_classes

        self.anomaly_criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([anomaly_pos_weight])
        )
        self.attribution_criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        anomaly_pred: torch.Tensor,
        attribution_pred: torch.Tensor,
        anomaly_labels: torch.Tensor,
        attribution_labels: torch.Tensor,
        telegrapher_loss: torch.Tensor,
        euler_bernoulli_loss: torch.Tensor,
        causality_loss: torch.Tensor,
    ) -> dict:
        """Compute combined loss.

        Args:
            anomaly_pred: Raw logits, shape (N, 1).
            attribution_pred: Class logits, shape (N, n_classes).
            anomaly_labels: Binary ground truth, shape (N,).
            attribution_labels: Integer class labels, shape (N,).
            telegrapher_loss: PDE residual scalar.
            euler_bernoulli_loss: PDE residual scalar.
            causality_loss: Causality constraint scalar.

        Returns:
            Dict with keys 'total', 'anomaly', 'attribution',
            'telegrapher', 'euler_bernoulli', 'causality'.
        """
        self.anomaly_criterion.pos_weight = self.anomaly_criterion.pos_weight.to(
            anomaly_pred.device
        )

        l_anom = self.anomaly_criterion(
            anomaly_pred.squeeze(), anomaly_labels.float()
        )
        l_attr = self.attribution_criterion(
            attribution_pred, attribution_labels.long()
        )
        l_total = (
            l_anom
            + l_attr
            + self.lambda_telegrapher * telegrapher_loss
            + self.lambda_euler_bernoulli * euler_bernoulli_loss
            + self.lambda_causality * causality_loss
        )
        return {
            "total": l_total,
            "anomaly": l_anom,
            "attribution": l_attr,
            "telegrapher": telegrapher_loss,
            "euler_bernoulli": euler_bernoulli_loss,
            "causality": causality_loss,
        }

    def __call__(self, *args, **kwargs) -> dict:
        return self.forward(*args, **kwargs)
