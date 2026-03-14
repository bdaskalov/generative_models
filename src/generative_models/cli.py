"""Command-line entry point for training generative models."""

from __future__ import annotations

import argparse
import sys

from generative_models.data.datamodule import ImageDataModule
from generative_models.models.vae import VAE
from generative_models.training.train import train

MODEL_REGISTRY: dict[str, type] = {
    "vae": VAE,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a deep generative model on image data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--model", choices=list(MODEL_REGISTRY), default="vae",
        help="Which generative model to train.",
    )

    data = p.add_argument_group("data")
    data.add_argument("--dataset", default="mnist", help="Dataset name (see DATASET_REGISTRY).")
    data.add_argument("--data-dir", default="./data", help="Root dir for dataset downloads.")
    data.add_argument("--image-size", type=int, default=32)
    data.add_argument("--n-channels", type=int, default=1, help="1=grayscale, 3=RGB.")
    data.add_argument("--batch-size", type=int, default=128)
    data.add_argument("--num-workers", type=int, default=4)
    data.add_argument("--val-fraction", type=float, default=0.1)

    model_g = p.add_argument_group("model (VAE)")
    model_g.add_argument("--latent-dim", type=int, default=64)
    model_g.add_argument("--hidden-dims", type=int, nargs="+", default=[32, 64, 128])
    model_g.add_argument("--kl-weight", type=float, default=1.0, help="Beta for beta-VAE.")

    train_g = p.add_argument_group("training")
    train_g.add_argument("--lr", type=float, default=1e-3)
    train_g.add_argument("--max-epochs", type=int, default=20)
    train_g.add_argument("--accelerator", default="auto")
    train_g.add_argument(
        "--precision", default="bf16-mixed",
        choices=["32", "16-mixed", "bf16-mixed"],
        help="Training precision. Mixed modes use tensor cores for matmuls/convs.",
    )
    train_g.add_argument("--log-dir", default="runs")
    train_g.add_argument("--output-dir", default="outputs", help="Dir for sample PNG grids.")
    train_g.add_argument("--sample-every-n-epochs", type=int, default=5)
    train_g.add_argument("--n-samples", type=int, default=16)

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    dm = ImageDataModule(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        image_size=args.image_size,
        n_channels=args.n_channels,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
    )

    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls(
        in_channels=args.n_channels,
        latent_dim=args.latent_dim,
        image_size=args.image_size,
        hidden_dims=args.hidden_dims,
        kl_weight=args.kl_weight,
        lr=args.lr,
    )

    train(
        model,
        datamodule=dm,
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        precision=args.precision,
        log_dir=args.log_dir,
        output_dir=args.output_dir,
        sample_every_n_epochs=args.sample_every_n_epochs,
        n_samples=args.n_samples,
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
