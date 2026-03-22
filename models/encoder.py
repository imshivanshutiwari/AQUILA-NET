import torch
import torch.nn as nn


class CableEncoder(nn.Module):
    """Autoencoder for learning compact cable state representations.

    Architecture
    ------------
    Encoder:  Linear(input_dim → 128) → LayerNorm → ReLU
              → Linear(128 → 64) → ReLU
              → Linear(64 → latent_dim)

    Decoder:  Linear(latent_dim → 64) → ReLU
              → Linear(64 → 128) → ReLU
              → Linear(128 → input_dim)
    """

    def __init__(self, input_dim: int, latent_dim: int = 64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map input to latent space.

        Args:
            x: tensor of shape (N, input_dim).

        Returns:
            Latent code of shape (N, latent_dim).
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct input from latent code.

        Args:
            z: latent tensor of shape (N, latent_dim).

        Returns:
            Reconstructed tensor of shape (N, input_dim).
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple:
        """Full autoencoder pass.

        Args:
            x: input tensor of shape (N, input_dim).

        Returns:
            Tuple (z, x_reconstructed) where z has shape (N, latent_dim) and
            x_reconstructed has shape (N, input_dim).
        """
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return z, x_reconstructed

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute mean-squared reconstruction error.

        Args:
            x: input tensor of shape (N, input_dim).

        Returns:
            Scalar MSE loss.
        """
        _, x_hat = self.forward(x)
        return nn.functional.mse_loss(x_hat, x)
