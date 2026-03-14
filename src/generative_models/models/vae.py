"""Simple convolutional Variational Autoencoder."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from generative_models.models.base import GenerativeModel


class VAE(GenerativeModel):
    """Convolutional VAE with a symmetric encoder/decoder.

    Parameters
    ----------
    in_channels:
        Number of input image channels (1 for grayscale, 3 for RGB).
    latent_dim:
        Dimensionality of the latent space.
    image_size:
        Spatial size the input images are resized to (must match the
        ``image_size`` used by the data module).
    hidden_dims:
        Channel counts for successive conv layers in the encoder; the
        decoder mirrors them in reverse.
    kl_weight:
        Multiplier on the KL term (beta-VAE when != 1).
    lr:
        Learning rate for Adam.
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 64,
        image_size: int = 32,
        hidden_dims: list[int] | None = None,
        kl_weight: float = 1.0,
        lr: float = 1e-3,
    ) -> None:
        super().__init__(lr=lr)
        self.save_hyperparameters()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.kl_weight = kl_weight
        self.in_channels = in_channels

        if hidden_dims is None:
            hidden_dims = [32, 64, 128]
        self.hidden_dims = hidden_dims

        self.encoder = self._build_encoder(in_channels, hidden_dims)

        # Run a dummy input through the encoder to get the flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size, image_size)
            dummy_out = self.encoder(dummy)
            self._enc_spatial = dummy_out.shape[2]
            self._enc_flat = dummy_out.numel()

        self.fc_mu = nn.Linear(self._enc_flat, latent_dim)
        self.fc_logvar = nn.Linear(self._enc_flat, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self._enc_flat)

        self.decoder = self._build_decoder(in_channels, hidden_dims)

    @staticmethod
    def _build_encoder(in_channels: int, hidden_dims: list[int]) -> nn.Sequential:
        layers: list[nn.Module] = []
        ch = in_channels
        for h_dim in hidden_dims:
            layers.extend([
                nn.Conv2d(ch, h_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU(),
            ])
            ch = h_dim
        return nn.Sequential(*layers)

    @staticmethod
    def _build_decoder(in_channels: int, hidden_dims: list[int]) -> nn.Sequential:
        layers: list[nn.Module] = []
        reversed_dims = list(reversed(hidden_dims))
        for i in range(len(reversed_dims) - 1):
            layers.extend([
                nn.ConvTranspose2d(
                    reversed_dims[i], reversed_dims[i + 1],
                    kernel_size=3, stride=2, padding=1, output_padding=1,
                ),
                nn.BatchNorm2d(reversed_dims[i + 1]),
                nn.LeakyReLU(),
            ])
        layers.extend([
            nn.ConvTranspose2d(
                reversed_dims[-1], in_channels,
                kernel_size=3, stride=2, padding=1, output_padding=1,
            ),
            nn.Sigmoid(),
        ])
        return nn.Sequential(*layers)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.encoder(x).flatten(start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        h = self.fc_decode(z)
        h = h.view(-1, self.hidden_dims[-1], self._enc_spatial, self._enc_spatial)
        out = self.decoder(h)
        return out[:, :, :self.image_size, :self.image_size]

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return {"x_hat": x_hat, "mu": mu, "logvar": logvar, "z": z}

    def compute_loss(self, batch: Tensor, fwd: dict[str, Tensor]) -> dict[str, Tensor]:
        x_hat, mu, logvar = fwd["x_hat"], fwd["mu"], fwd["logvar"]
        # Sum over pixels per sample, then average over the batch — keeps
        # reconstruction and KL on the same per-sample scale and avoids
        # the posterior collapse caused by mean-reduced MSE.
        recon = F.binary_cross_entropy(x_hat, batch, reduction="sum") / batch.shape[0]
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch.shape[0]
        loss = recon + self.kl_weight * kl
        return {"loss": loss, "recon_loss": recon, "kl_loss": kl}

    def sample(self, n: int) -> Tensor:
        z = torch.randn(n, self.latent_dim, device=self.device)
        return self.decode(z)
