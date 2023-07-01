import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataloader.loader import *


class VocDataset(pl.LightningDataModule):
    DEFAULT_TARGET_CLASSES = ['person']

    def __init__(self, train_batch=4, val_batch=8, target_classes=None):
        super().__init__()
        if target_classes is None:
            target_classes = VocDataset.DEFAULT_TARGET_CLASSES
        self._target_classes = target_classes
        self._train_batch = train_batch
        self._val_batch = val_batch

    def train_dataloader(self) -> DataLoader:
        return build_dataloader(
            subset='train',
            batch_size=self._train_batch,
            shuffle=True,
            download=False,
            root="vocdata",
            target_classes=self._target_classes,
        )

    def val_dataloader(self) -> DataLoader:
        return build_dataloader(
            subset='val',
            batch_size=self._val_batch,
            shuffle=False,
            download=False,
            root="vocdata",
            target_classes=self._target_classes,
        )
