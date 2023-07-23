import numpy as np
import pytorch_lightning as pl
import torch
import torchvision

from model import DETR
from dataloader.visualization import visualize_batch
from dataloader.voc_labels import VocLabelsCodec
from util.bbox import denormalize_boxes

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
        if torch.sum(obj_cnt) == 0:
            return None

        targets, offset = [], 0
        for cnt in obj_cnt:
            img_targets = {
                'labels': labels[offset:offset + cnt],
                'boxes': boxes[offset:offset + cnt],
            }
            targets.append(img_targets)
            offset += cnt

        predictions = self.model(images, targets)

        scalars = {
            f"loss/total": predictions['loss'].detach(),
            f"loss/cardinality": predictions['cardinality_error'],
            f"loss/boxes": predictions['regression'].detach(),
            f"loss/labels": predictions['classification'].detach(),
        }
        self.log_dict(scalars)

        writer = self.logger.experiment
        for k, v in scalars.items():
            writer.add_scalar(k, v, self.global_step)

        return predictions


    def validation_step(self, batch, batch_idx: int) -> None:
        if batch_idx == 0:
            images, boxes, labels, obj_cnt = batch
            predictions = self.model(images)

            gt_boxes_list, gt_labels_list, offset = [], [], 0
            for cnt in obj_cnt:
                gt_boxes_list.append(boxes[offset:offset + cnt])
                gt_labels_list.append(labels[offset:offset + cnt])
                offset += cnt

            img_shapes = [x.shape for x in images]
            denormalized_boxes = denormalize_boxes(gt_boxes_list, img_shapes)

            self.visualize_target(
                images=images,
                labels=gt_labels_list,
                boxes=denormalized_boxes,
            )

            pr_labels_list, pr_boxes_list, pr_scores_list = [], [], []
            for boxes, scores in zip(predictions['boxes'], predictions['scores']):
                max_scores, _ = torch.max(scores, dim=1)
                labels = torch.argmax(scores, dim=-1)
                pr_labels_list.append(labels)
                pr_boxes_list.append(boxes)
                pr_scores_list.append(max_scores)

            self.visualize_prediction(
                images=images,
                labels=pr_labels_list,
                scores=pr_scores_list,
                boxes=pr_boxes_list,
            )

    def visualize_target(self, images, labels, boxes):
        codec = VocLabelsCodec(['person'])
        visualizations = visualize_batch(
            images,
            boxes_batch=boxes,
            labels_batch=labels,
            scores_batch=None,
            codec=codec,
            return_images=True
        )
        image_tensors = [torch.permute(torch.from_numpy(x), (2, 0, 1)) for x in visualizations]
        pred_grid = torchvision.utils.make_grid(image_tensors)
        writer = self.logger.experiment
        writer.add_image(f'Targets', pred_grid, self.global_step)

    def visualize_prediction(self, images, labels, scores, boxes):
        codec = VocLabelsCodec(['person'])
        visualizations = visualize_batch(
            images,
            boxes_batch=boxes,
            labels_batch=labels,
            scores_batch=scores,
            codec=codec,
            return_images=True
        )
        image_tensors = [torch.permute(torch.from_numpy(x), (2, 0, 1)) for x in visualizations]
        pred_grid = torchvision.utils.make_grid(image_tensors)
        writer = self.logger.experiment
        writer.add_image(f'Prediction', pred_grid, self.global_step)
