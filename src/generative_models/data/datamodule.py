"""Lightning DataModule for torchvision image datasets."""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

DATASET_REGISTRY: dict[str, type[datasets.VisionDataset]] = {
    "mnist": datasets.MNIST,
    "fashion_mnist": datasets.FashionMNIST,
    "kmnist": datasets.KMNIST,
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
    "svhn": datasets.SVHN,
    "celeba": datasets.CelebA,
}


def _default_transform(image_size: int, n_channels: int) -> transforms.Compose:
    steps: list[Any] = []
    steps.append(transforms.Resize(image_size))
    steps.append(transforms.CenterCrop(image_size))
    if n_channels == 1:
        steps.append(transforms.Grayscale(num_output_channels=1))
    steps.append(transforms.ToTensor())
    return transforms.Compose(steps)


class ImageDataModule(L.LightningDataModule):
    """Wraps any torchvision dataset behind a uniform Lightning interface.

    Parameters
    ----------
    dataset_name:
        Key into ``DATASET_REGISTRY`` (case-insensitive) **or** a fully-qualified
        class path such as ``"torchvision.datasets.EMNIST"``.
    data_dir:
        Root directory where dataset files are downloaded / cached.
    image_size:
        Spatial size that every image is resized+cropped to.
    n_channels:
        1 for grayscale, 3 for RGB.  Grayscale conversion is applied when
        ``n_channels=1`` and the source images are RGB.
    batch_size:
        Mini-batch size used by all dataloaders.
    num_workers:
        Dataloader worker processes.
    val_fraction:
        Fraction of the training set held out for validation.
    dataset_kwargs:
        Extra keyword arguments forwarded to the dataset constructor (e.g.
        ``split="train"`` for SVHN, ``target_type="attr"`` for CelebA).
    """

    def __init__(
        self,
        dataset_name: str = "mnist",
        data_dir: str = "./data",
        image_size: int = 32,
        n_channels: int = 1,
        batch_size: int = 128,
        num_workers: int = 4,
        val_fraction: float = 0.1,
        **dataset_kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.dataset_cls = self._resolve_dataset(dataset_name)
        self.data_dir = data_dir
        self.image_size = image_size
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.dataset_kwargs = dataset_kwargs

        self.transform = _default_transform(image_size, n_channels)
        self.ds_train: torch.utils.data.Dataset | None = None
        self.ds_val: torch.utils.data.Dataset | None = None
        self.ds_test: torch.utils.data.Dataset | None = None

    @staticmethod
    def _resolve_dataset(name: str) -> type[datasets.VisionDataset]:
        key = name.lower().replace("-", "_")
        if key in DATASET_REGISTRY:
            return DATASET_REGISTRY[key]
        import importlib

        module_path, cls_name = name.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, cls_name)

    def _make_dataset(self, train: bool) -> datasets.VisionDataset:
        cls = self.dataset_cls
        # SVHN uses split= instead of train=
        if cls is datasets.SVHN:
            return cls(
                root=self.data_dir,
                split="train" if train else "test",
                transform=self.transform,
                download=True,
                **self.dataset_kwargs,
            )
        return cls(
            root=self.data_dir,
            train=train,
            transform=self.transform,
            download=True,
            **self.dataset_kwargs,
        )

    def setup(self, stage: str | None = None) -> None:
        if stage in ("fit", None):
            full_train = self._make_dataset(train=True)
            n_val = int(len(full_train) * self.val_fraction)
            n_train = len(full_train) - n_val
            self.ds_train, self.ds_val = random_split(
                full_train,
                [n_train, n_val],
                generator=torch.Generator().manual_seed(42),
            )
        if stage in ("test", None):
            self.ds_test = self._make_dataset(train=False)

    def train_dataloader(self) -> DataLoader:
        assert self.ds_train is not None
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.ds_val is not None
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.ds_test is not None
        return DataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
