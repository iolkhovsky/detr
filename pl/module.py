import numpy as np
import pytorch_lightning as pl
import torch

from model import DETR


class DetrModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.model = DETR(*args, **kwargs)

    def configure_optimizers(self):
        param_dicts = [
            {
                'params': [
                    p for n, p in self.model.named_parameters()
                    if 'backbone' not in n and p.requires_grad
                ]
            },
            {
                'params': [
                    p for n, p in self.model.named_parameters()
                    if 'backbone' in n and p.requires_grad
                ],
                'lr': 1e-5,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=1e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 200)
        return [[optimizer], [scheduler]]

    def configure_callbacks(self):
        pass

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        images, boxes, labels, obj_cnt = batch
        targets, offset = [], 0
        for cnt in obj_cnt:
            img_targets = {
                'labels': labels[offset:offset + cnt],
                'boxes': boxes[offset:offset + cnt],
            }
            targets.append(img_targets)
            offset += cnt

        predictions = self.model(images, targets)
        # self.log_dict(
        #     {
        #         f"loss/train": loss.detach(),
        #         f"accuracy/train": acc,
        #     }
        # )
        return predictions['loss']


    def validation_step(self, batch, batch_idx: int) -> None:
      pass
