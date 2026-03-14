"""Model architectures for generative models (VAE, GAN, Diffusion, etc.)."""

from generative_models.models.base import GenerativeModel
from generative_models.models.vae import VAE

__all__ = ["GenerativeModel", "VAE"]
