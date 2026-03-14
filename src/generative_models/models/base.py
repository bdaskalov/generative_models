"""Base class for all generative models in this project."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import lightning as L
import torch
from torch import Tensor


class GenerativeModel(L.LightningModule):
    """Abstract base for generative models with built-in metric logging.

    Subclasses must implement ``forward``, ``compute_loss``, and ``sample``.
    The base class takes care of the train/val/test step boilerplate and logs
    every key returned by ``compute_loss`` under a ``{stage}/{key}`` namespace
    so that different models can be compared side-by-side in TensorBoard.
    """

    def __init__(self, lr: float = 1e-3) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

    @abstractmethod
    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Run a full encode-decode (or generator) pass.

        Must return a dict that at least contains ``"x_hat"`` (the
        reconstruction / generation).  Additional entries (e.g. ``"mu"``,
        ``"logvar"``) are passed through to ``compute_loss``.
        """

    @abstractmethod
    def compute_loss(self, batch: Tensor, fwd: dict[str, Tensor]) -> dict[str, Tensor]:
        """Return a dict of named loss components.

        The dict **must** include a ``"loss"`` key with the scalar used for
        back-propagation.  All other keys are logged as metrics.
        """

    @abstractmethod
    def sample(self, n: int) -> Tensor:
        """Generate ``n`` new images from the prior."""

    def _shared_step(self, batch: Any, stage: str) -> Tensor:
        x, *_ = batch
        fwd = self(x)
        losses = self.compute_loss(x, fwd)
        for name, value in losses.items():
            self.log(
                f"{stage}/{name}",
                value,
                on_step=(stage == "train"),
                on_epoch=True,
                prog_bar=(name == "loss"),
                logger=True,
            )
        return losses["loss"]

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int) -> Tensor:
        return self._shared_step(batch, "test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)
