import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataloader.loader import build_dataloader


class VocDataset(pl.LightningDataModule):
    DEFAULT_TARGET_CLASSES = ['person']

    def __init__(self, train_batch=4, val_batch=8, target_classes=None, download=False):
        super().__init__()
        if target_classes is None:
            target_classes = VocDataset.DEFAULT_TARGET_CLASSES
        self._target_classes = target_classes
        self._train_batch = train_batch
        self._val_batch = val_batch
        self._download = download

    def train_dataloader(self) -> DataLoader:
        return build_dataloader(
            subset='train',
            batch_size=self._train_batch,
            shuffle=True,
            download=self._download,
            root="vocdata",
            target_classes=self._target_classes,
        )

    def val_dataloader(self) -> DataLoader:
        return build_dataloader(
            subset='val',
            batch_size=self._val_batch,
            shuffle=False,
            download=self._download,
            root="vocdata",
            target_classes=self._target_classes,
            max_size=128,
        )
