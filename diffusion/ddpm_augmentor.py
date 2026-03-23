import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DDPMAugmentor(nn.Module):
    """DDPM with T=1000, linear beta schedule, conditioned on cable parameters.

    Architecture: 1-D UNet with 3 encoder / decoder levels, sinusoidal time
    embedding, and a cross-attention bottleneck that injects the cable
    condition.
    """

    def __init__(
        self,
        seq_len: int = 1000,
        condition_dim: int = 32,
        T: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.condition_dim = condition_dim
        self.T = T

        # ── Noise schedule ──────────────────────────────────────────────────
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

        # ── Time embedding: sinusoidal → MLP ────────────────────────────────
        self.time_embed = nn.Sequential(
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
        )

        # ── Condition projection (condition_dim → 256) ──────────────────────
        self.cond_proj = nn.Linear(condition_dim, 256)

        # ── Encoder ─────────────────────────────────────────────────────────
        self.enc1 = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(64, 128, 3, padding=1, stride=2),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(128, 256, 3, padding=1, stride=2),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
        )

        # ── Bottleneck cross-attention ───────────────────────────────────────
        self.bottleneck_attn = nn.MultiheadAttention(256, 4, batch_first=True)

        # ── Decoder ─────────────────────────────────────────────────────────
        # dec3 receives cat(h, e3) → 512 ch, upsamples × 2 → 128 ch
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(512, 128, 3, padding=1, stride=2, output_padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
        )
        # dec2 receives cat(d3, e2) → 256 ch, upsamples × 2 → 64 ch
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(256, 64, 3, padding=1, stride=2, output_padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        # dec1 receives cat(d2, e1) → 128 ch → 1 ch
        self.dec1 = nn.Conv1d(128, 1, 3, padding=1)

    # ── Sinusoidal time embedding ────────────────────────────────────────────
    def _sinusoidal_embedding(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        """Positional sinusoidal embedding for diffusion timesteps."""
        half_dim = dim // 2
        emb_scale = math.log(10000.0) / (half_dim - 1)
        freqs = torch.exp(
            torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb_scale
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half_dim)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)   # (B, dim)

    # ── Forward (noise prediction) ───────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict the noise added to x at timestep t.

        Args:
            x:         (B, 1, L)
            t:         (B,) integer timesteps
            condition: (B, condition_dim) or None

        Returns:
            Predicted noise tensor of shape (B, 1, L).
        """
        # Time embedding
        t_emb = self._sinusoidal_embedding(t, 256)  # (B, 256)
        t_emb = self.time_embed(t_emb)               # (B, 256)

        # Condition embedding
        if condition is not None:
            cond_emb = self.cond_proj(condition) + t_emb  # (B, 256)
        else:
            cond_emb = t_emb                               # (B, 256)

        # ── Encoder ─────────────────────────────────────────────────────────
        e1 = self.enc1(x)   # (B, 64,  L)
        e2 = self.enc2(e1)  # (B, 128, L//2)
        e3 = self.enc3(e2)  # (B, 256, L//4)

        # ── Bottleneck cross-attention ───────────────────────────────────────
        B, C, L4 = e3.shape
        q = e3.permute(0, 2, 1)                              # (B, L//4, 256)
        kv = cond_emb.unsqueeze(1)                           # (B, 1,    256)
        attn_out, _ = self.bottleneck_attn(q, kv, kv)       # (B, L//4, 256)
        h = attn_out.permute(0, 2, 1) + e3                  # (B, 256,  L//4)

        # ── Decoder ─────────────────────────────────────────────────────────
        d3_in = torch.cat([h, e3], dim=1)                   # (B, 512,  L//4)
        d3 = self.dec3(d3_in)                               # (B, 128,  ~L//2)
        if d3.shape[-1] != e2.shape[-1]:
            d3 = F.interpolate(d3, size=e2.shape[-1], mode="linear", align_corners=False)

        d2_in = torch.cat([d3, e2], dim=1)                  # (B, 256,  L//2)
        d2 = self.dec2(d2_in)                               # (B, 64,   ~L)
        if d2.shape[-1] != e1.shape[-1]:
            d2 = F.interpolate(d2, size=e1.shape[-1], mode="linear", align_corners=False)

        d1_in = torch.cat([d2, e1], dim=1)                  # (B, 128,  L)
        out = self.dec1(d1_in)                              # (B, 1,    L)
        return out

    # ── Diffusion helpers ────────────────────────────────────────────────────
    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward (noising) process: x_t = sqrt(ᾱ_t)*x0 + sqrt(1-ᾱ_t)*noise."""
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self.alpha_bars[t].to(x0.device)             # (B,)
        # Reshape for broadcasting with (B, 1, L)
        ab = ab.view(-1, 1, 1)
        return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise

    def p_losses(
        self,
        x0: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Training loss: MSE between predicted and actual noise."""
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred = self.forward(xt, t, condition)
        return F.mse_loss(pred, noise)

    # ── Sampling ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate_sabotage_samples(
        self,
        n_samples: int = 5000,
        cable_material: int = 0,
        depth_m: float = 3500.0,
        age_years: float = 15.0,
        seq_len: Optional[int] = None,
    ) -> np.ndarray:
        """Generate synthetic sabotage OTDR traces via DDPM reverse diffusion.

        Processes *n_samples* in mini-batches of 50 for memory efficiency.

        Returns:
            numpy array of shape (n_samples, seq_len).
        """
        self.eval()
        seq_len = seq_len or self.seq_len
        device = next(self.parameters()).device
        batch_size = 50
        all_samples = []

        for start in range(0, n_samples, batch_size):
            bs = min(batch_size, n_samples - start)

            # Build conditioning tensor
            mat_t = torch.full((bs,), cable_material, dtype=torch.long, device=device)
            dep_t = torch.full((bs,), depth_m, device=device)
            age_t = torch.full((bs,), age_years, device=device)
            sev_t = torch.zeros(bs, device=device)
            pos_t = torch.zeros(bs, device=device)

            # Inline import to avoid circular deps at module level
            from diffusion.cable_conditioner import CableConditioner

            # Attempt to reuse a CableConditioner if stored, otherwise build ad-hoc
            if not hasattr(self, "_conditioner"):
                conditioner = CableConditioner(self.condition_dim).to(device)
                conditioner.eval()
            else:
                conditioner = self._conditioner

            condition = conditioner(mat_t, dep_t, age_t, sev_t, pos_t)  # (bs, cond_dim)

            # Start from pure Gaussian noise
            x = torch.randn(bs, 1, seq_len, device=device)

            # DDPM reverse process
            for step in reversed(range(self.T)):
                t_batch = torch.full((bs,), step, dtype=torch.long, device=device)
                pred_noise = self.forward(x, t_batch, condition)

                alpha = self.alphas[step]
                alpha_bar = self.alpha_bars[step]
                beta = self.betas[step]

                # Mean prediction
                coeff = (1.0 - alpha) / (1.0 - alpha_bar).sqrt()
                x_mean = (x - coeff * pred_noise) / alpha.sqrt()

                if step > 0:
                    noise = torch.randn_like(x)
                    x = x_mean + beta.sqrt() * noise
                else:
                    x = x_mean

            all_samples.append(x.squeeze(1).cpu().numpy())  # (bs, seq_len)

        return np.concatenate(all_samples, axis=0)  # (n_samples, seq_len)
