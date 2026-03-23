import math
from typing import Optional

import torch


class PINNResidual:
    """Computes physics-informed residuals for the Telegrapher and Euler-Bernoulli PDEs.

    Derivatives are computed using ``torch.autograd.grad`` so that both
    residuals and their gradients are available for PINN training.
    """

    def __init__(
        self,
        R: float = 0.034,
        L: float = 0.0002,
        C: float = 0.00015,
        G: float = 0.000001,
        EI: float = 2.5e8,
    ):
        self.R = R
        self.L = L
        self.C = C
        self.G = G
        self.EI = EI
        self.rho_A = 8900.0 * math.pi * 0.02 ** 2

    # ------------------------------------------------------------------
    # Telegrapher's equations
    # ------------------------------------------------------------------

    def telegrapher_residual(
        self,
        V: torch.Tensor,
        I: torch.Tensor,
        x_coords: torch.Tensor,
        t_coords: torch.Tensor,
    ) -> float:
        """Compute the mean Telegrapher PDE residual.

        Telegrapher equations:
            ∂V/∂x + L·∂I/∂t + R·I = 0
            ∂I/∂x + C·∂V/∂t + G·V = 0

        Args:
            V: voltage values at collocation points, shape (N,); must require grad.
            I: current values at collocation points, shape (N,); must require grad.
            x_coords: spatial coordinates, shape (N,); must require grad.
            t_coords: temporal coordinates, shape (N,); must require grad.

        Returns:
            Scalar float – mean squared residual of both equations.
        """
        def _grad(outputs, inputs):
            return torch.autograd.grad(
                outputs,
                inputs,
                grad_outputs=torch.ones_like(outputs),
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]

        dV_dx = _grad(V, x_coords)
        dI_dt = _grad(I, t_coords)
        dI_dx = _grad(I, x_coords)
        dV_dt = _grad(V, t_coords)

        # Replace None gradients with zeros
        if dV_dx is None:
            dV_dx = torch.zeros_like(V)
        if dI_dt is None:
            dI_dt = torch.zeros_like(I)
        if dI_dx is None:
            dI_dx = torch.zeros_like(I)
        if dV_dt is None:
            dV_dt = torch.zeros_like(V)

        residual_V = dV_dx + self.L * dI_dt + self.R * I
        residual_I = dI_dx + self.C * dV_dt + self.G * V

        mse = ((residual_V ** 2 + residual_I ** 2) / 2.0).mean()
        return float(mse.item())

    # ------------------------------------------------------------------
    # Euler-Bernoulli beam equation
    # ------------------------------------------------------------------

    def euler_bernoulli_residual(
        self,
        w: torch.Tensor,
        x_coords: torch.Tensor,
        t_coords: torch.Tensor,
        q: Optional[torch.Tensor] = None,
    ) -> float:
        """Compute the mean Euler-Bernoulli PDE residual.

        PDE: EI·∂⁴w/∂x⁴ + ρA·∂²w/∂t² = q(x, t)

        Higher-order spatial derivatives are computed by repeated application
        of ``torch.autograd.grad``.

        Args:
            w: transverse displacement values, shape (N,); must require grad.
            x_coords: spatial coordinates, shape (N,); must require grad.
            t_coords: temporal coordinates, shape (N,); must require grad.
            q: distributed load, shape (N,); defaults to zero (free vibration).

        Returns:
            Scalar float – mean squared residual.
        """
        def _grad(outputs, inputs):
            # If outputs has no connection to inputs, skip the grad call
            if outputs.grad_fn is None and not outputs.requires_grad:
                return inputs * 0.0  # zeros that stay in the computation graph
            g = torch.autograd.grad(
                outputs,
                inputs,
                grad_outputs=torch.ones_like(outputs),
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if g is None:
                return inputs * 0.0  # zeros that stay in the computation graph
            return g

        # Fourth spatial derivative via repeated differentiation
        dw_dx = _grad(w, x_coords)
        d2w_dx2 = _grad(dw_dx, x_coords)
        d3w_dx3 = _grad(d2w_dx2, x_coords)
        d4w_dx4 = _grad(d3w_dx3, x_coords)

        # Second temporal derivative
        dw_dt = _grad(w, t_coords)
        d2w_dt2 = _grad(dw_dt, t_coords)

        if q is None:
            q = torch.zeros_like(w)

        residual = self.EI * d4w_dx4 + self.rho_A * d2w_dt2 - q
        mse = (residual ** 2).mean()
        return float(mse.item())
