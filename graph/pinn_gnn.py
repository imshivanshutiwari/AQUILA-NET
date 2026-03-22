import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GATv2Conv

from .message_passing import PhysicsConstrainedMPNN


class PINNGraphNeuralNetwork(nn.Module):
    """Physics-Informed Neural Network over a cable graph.

    Combines physics-constrained message passing (Telegrapher / Euler-Bernoulli)
    with graph attention for anomaly detection and cause attribution.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int = 6,
        hidden_dim: int = 256,
        n_classes: int = 4,
        dropout: float = 0.1,
        R: float = 0.034,
        L: float = 0.0002,
        C: float = 0.00015,
        G: float = 0.000001,
        EI: float = 2.5e8,
        rho_A: float = None,
    ):
        super().__init__()

        # Physical parameters
        self.R = R
        self.L = L
        self.C = C
        self.G = G
        self.EI = EI
        self.rho_A = rho_A if rho_A is not None else 8900.0 * math.pi * 0.02 ** 2
        self.wave_speed = 1.0 / math.sqrt(L * C)
        self.dropout = dropout

        # Graph convolution layers
        self.conv1 = PhysicsConstrainedMPNN(node_dim, 256, R=R, L=L, C=C)
        self.conv2 = PhysicsConstrainedMPNN(256, 128, R=R, L=L, C=C)
        # GATv2Conv with concat=True and heads=4: output dim = 64 * 4 = 256
        self.conv3 = GATv2Conv(128, 64, heads=4, dropout=dropout, concat=True)

        # Anomaly detection head
        self.anomaly_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Attribution classification head
        self.attribution_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
            nn.Softmax(dim=-1),
        )

    # ------------------------------------------------------------------

    def forward(self, data) -> tuple:
        """Forward pass returning (anomaly_scores, attribution_probs).

        Args:
            data: torch_geometric.data.Data with x and edge_index attributes.

        Returns:
            Tuple of:
                anomaly   – shape (N, 1), values in [0, 1].
                attribution – shape (N, n_classes), sums to 1 per row.
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = (
            data.edge_attr
            if hasattr(data, "edge_attr") and data.edge_attr is not None
            else None
        )

        h1 = self.conv1(x, edge_index, edge_attr)
        h2 = self.conv2(h1, edge_index, edge_attr)
        h3 = self.conv3(h2, edge_index)
        h3 = F.dropout(h3, p=self.dropout, training=self.training)

        anomaly = self.anomaly_head(h3)
        attribution = self.attribution_head(h3)
        return anomaly, attribution

    # ------------------------------------------------------------------
    # Physics loss terms
    # ------------------------------------------------------------------

    def compute_telegrapher_loss(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Approximate Telegrapher PDE residual.

        PDE: ∂²V/∂x² - LC·∂²V/∂t² - (RC+GL)·∂V/∂t - RG·V = 0

        Args:
            x: node features, shape (N, F).
            edge_index: shape (2, E).
            edge_attr: shape (E, A); first column is length_km.

        Returns:
            Scalar mean-squared residual.
        """
        row, col = edge_index

        if edge_attr is not None:
            dx = edge_attr[:, 0].clamp(min=0.1) * 1000.0  # m
        else:
            dx = torch.ones(row.shape[0], device=x.device) * 1000.0

        V_i = x[row, 0]
        V_j = x[col, 0]
        d2V_dx2 = (V_j - 2 * V_i + V_j) / (dx ** 2)

        # Channel 1 used as dV/dt proxy
        dVdt = x[row, 1 % x.shape[1]]
        V = x[row, 0]

        LC = self.L * self.C
        RC_GL = self.R * self.C + self.G * self.L
        RG = self.R * self.G

        residual = d2V_dx2 - LC * dVdt - RC_GL * dVdt - RG * V
        return (residual ** 2).mean()

    def compute_euler_bernoulli_loss(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Approximate Euler-Bernoulli PDE residual.

        PDE: EI·∂⁴w/∂x⁴ + ρA·∂²w/∂t² = q(x, t)

        Args:
            x: node features, shape (N, F).
            edge_index: shape (2, E).
            edge_attr: shape (E, A); first column is length_km.

        Returns:
            Scalar mean-squared residual.
        """
        row, col = edge_index

        if edge_attr is not None:
            dx = edge_attr[:, 0].clamp(min=0.1) * 1000.0
        else:
            dx = torch.ones(row.shape[0], device=x.device) * 1000.0

        w = x[row, 0]
        d4w_dx4_approx = (x[col, 0] - 2 * w + x[row, 0]) / (dx ** 4)
        d2w_dt2 = x[row, 2 % x.shape[1]]
        q = torch.zeros_like(w)

        residual = self.EI * d4w_dx4_approx + self.rho_A * d2w_dt2 - q
        return (residual ** 2).mean()

    def compute_causality_loss(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Penalise violation of causal propagation speed.

        A score at destination should not exceed the source score beyond the
        causal propagation limit.

        Args:
            x: node features, shape (N, F).
            edge_index: shape (2, E).
            edge_attr: shape (E, A); first column is length_km.

        Returns:
            Scalar mean-squared violation.
        """
        row, col = edge_index

        if edge_attr is not None:
            distance_km = edge_attr[:, 0].clamp(min=1.0)
        else:
            distance_km = torch.ones(row.shape[0], device=x.device)

        score_i = x[row, 0]
        score_j = x[col, 0]
        causal_max = score_i * torch.exp(-distance_km / (self.wave_speed * 1000))
        violation = torch.relu(score_j - causal_max - 0.1)
        return (violation ** 2).mean()
