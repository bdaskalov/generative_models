"""Training entry point for generative models."""

from __future__ import annotations

import lightning as L

from generative_models.data.datamodule import ImageDataModule
from generative_models.models.base import GenerativeModel
from generative_models.training.callbacks import SampleGridCallback


def train(
    model: GenerativeModel,
    datamodule: ImageDataModule | None = None,
    max_epochs: int = 20,
    accelerator: str = "auto",
    log_dir: str = "runs",
    sample_every_n_epochs: int = 5,
    n_samples: int = 16,
) -> L.Trainer:
    """Train a generative model and return the trainer.

    Parameters
    ----------
    model:
        A ``GenerativeModel`` subclass (e.g. ``VAE``).
    datamodule:
        Data source; defaults to MNIST at 32 x 32 if *None*.
    max_epochs:
        Training duration.
    accelerator:
        ``"auto"``, ``"gpu"``, ``"cpu"``, etc.
    log_dir:
        Root directory for TensorBoard logs.  Each model class gets its
        own sub-directory so metrics are easy to compare.
    sample_every_n_epochs:
        How often to log a grid of generated samples.
    n_samples:
        Number of images in the sample grid.
    """
    if datamodule is None:
        datamodule = ImageDataModule()

    logger = L.pytorch.loggers.TensorBoardLogger(
        save_dir=log_dir,
        name=type(model).__name__,
    )

    callbacks: list[L.Callback] = [
        SampleGridCallback(
            every_n_epochs=sample_every_n_epochs,
            n_samples=n_samples,
        ),
        L.pytorch.callbacks.ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            filename="{epoch}-{val/loss:.4f}",
        ),
        L.pytorch.callbacks.LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        logger=logger,
        callbacks=callbacks,
        deterministic=True,
    )

    trainer.fit(model, datamodule=datamodule)
    return trainer
