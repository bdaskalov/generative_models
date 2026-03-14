"""Custom Lightning callbacks for generative model training."""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
from torchvision.utils import make_grid

from generative_models.models.base import GenerativeModel


class SampleGridCallback(L.Callback):
    """Log a grid of generated samples at fixed intervals."""

    def __init__(self, every_n_epochs: int = 5, n_samples: int = 16) -> None:
        self.every_n_epochs = every_n_epochs
        self.n_samples = n_samples

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not isinstance(pl_module, GenerativeModel):
            return
        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs != 0:
            return

        with torch.no_grad():
            samples = pl_module.sample(self.n_samples).float()
        grid = make_grid(samples, nrow=4, normalize=True, value_range=(0, 1))

        for logger in trainer.loggers:
            if isinstance(logger, L.pytorch.loggers.TensorBoardLogger):
                logger.experiment.add_image(
                    "samples",
                    grid,
                    global_step=trainer.global_step,
                )

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx != 0 or not isinstance(pl_module, GenerativeModel):
            return
        if trainer.current_epoch % self.every_n_epochs != 0:
            return

        x, *_ = batch
        with torch.no_grad():
            fwd = pl_module(x[: self.n_samples])

        originals = make_grid(
            x[: self.n_samples].float(), nrow=4, normalize=True, value_range=(0, 1),
        )
        recons = make_grid(
            fwd["x_hat"][: self.n_samples].float(), nrow=4, normalize=True, value_range=(0, 1),
        )

        for logger in trainer.loggers:
            if isinstance(logger, L.pytorch.loggers.TensorBoardLogger):
                step = trainer.global_step
                logger.experiment.add_image("reconstructions/input", originals, global_step=step)
                logger.experiment.add_image("reconstructions/output", recons, global_step=step)
