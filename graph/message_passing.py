import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class PhysicsConstrainedMPNN(MessagePassing):
    """Message Passing Neural Network with Telegrapher-equation causal weighting."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        R: float = 0.034,
        L: float = 0.0002,
        C: float = 0.00015,
    ):
        super().__init__(aggr="add")

        self.lin = nn.Linear(in_channels, out_channels)
        self.res_lin = nn.Linear(in_channels, out_channels) if in_channels != out_channels else None

        self.R = R
        self.L = L
        self.C = C
        self.wave_speed = 1.0 / math.sqrt(L * C)

        # Pre-compute a fixed projection weight for the residual connection
        # fallback path in update().  This is only used when in_channels ==
        # out_channels (in which case res_lin is None and shapes always match),
        # so the branch is effectively a safety net rather than a hot path.
        self._proj_weight: torch.Tensor = None

    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None,
    ) -> torch.Tensor:
        # Normalise: always pass edge_attr so PyG's generated code is satisfied.
        # A zero-distance dummy gives causal_weight ≈ 1 (no attenuation).
        if edge_attr is None:
            edge_attr = torch.zeros(
                edge_index.shape[1], 1, dtype=x.dtype, device=x.device
            )
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    # ------------------------------------------------------------------

    def message(
        self,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        distance_km = edge_attr[:, 0].clamp(min=1.0)
        causal_weight = torch.exp(-distance_km / (self.wave_speed * 1000))
        return self.lin(x_j) * causal_weight.unsqueeze(-1)

    # ------------------------------------------------------------------

    def update(
        self,
        aggr_out: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        res = self.res_lin(x) if self.res_lin is not None else x
        if res.shape[-1] != aggr_out.shape[-1]:
            # Lazy-init projection weight to avoid re-allocating on every call.
            out_dim, in_dim = aggr_out.shape[-1], x.shape[-1]
            if (
                self._proj_weight is None
                or self._proj_weight.shape != (out_dim, in_dim)
                or self._proj_weight.device != x.device
            ):
                self._proj_weight = torch.eye(out_dim, in_dim, device=x.device)
            res_proj = F.linear(x, self._proj_weight)
            return torch.relu(aggr_out + res_proj)
        return torch.relu(aggr_out + res)

    # ------------------------------------------------------------------

    def compute_pde_violation(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Approximate Telegrapher equation residual on graph edges.

        Args:
            x_i: source node features, shape (E, F).
            x_j: destination node features, shape (E, F).
            edge_attr: edge attributes, shape (E, A); first column is length_km.

        Returns:
            Scalar mean-squared residual.
        """
        if edge_attr is not None:
            distance_km = edge_attr[:, 0].clamp(min=0.1)
        else:
            distance_km = torch.ones(x_i.shape[0], device=x_i.device)

        dV_dx = (x_j[:, 0] - x_i[:, 0]) / (distance_km * 1000)

        if x_i.shape[1] > 1:
            dI_dt = (x_j[:, 1] - x_i[:, 1]) / 0.001
        else:
            dI_dt = torch.zeros_like(dV_dx)

        residual_V = dV_dx + self.L * dI_dt + self.R * x_i[:, 1 % x_i.shape[1]]
        return (residual_V ** 2).mean()
