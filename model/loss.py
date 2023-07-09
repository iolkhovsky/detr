import torch
import torch.nn as nn
import torch.nn.functional as F

from model.matcher import HungarianMatcher
from util.bbox import generalized_box_iou


class RegressionLoss(nn.Module):
    def __init__(self, l1_weight=None, giou_weight=None):
        super().__init__()
        self._l1_w = l1_weight
        self._giou_w = giou_weight

    def forward(self, predictions, targets):
        if len(targets) == 0:
            return 0.

        loss_bbox = F.l1_loss(predictions, targets, reduction='none')
        num_boxes = len(targets)

        l1_loss = loss_bbox.sum() / num_boxes
        if self._l1_w:
            l1_loss = l1_loss * self._l1_w

        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                predictions,
                targets,
            )
        )
        loss_giou = loss_giou.sum()
        if self._giou_w:
            loss_giou = loss_giou * self._giou_w

        return l1_loss + loss_giou


class ClassificationLoss(nn.Module):
    def __init__(self, weight=None, num_classes=1, eos_coef=0.1):
        super().__init__()
        self._w = weight
        self._empty_weight = torch.ones(num_classes + 1)
        self._empty_weight[0] = eos_coef
        self.register_buffer('empty_weight', self._empty_weight)

    def forward(self, logits, targets):
        self._empty_weight = self._empty_weight.to(logits.device)
        ce = F.cross_entropy(logits, targets.long(), self._empty_weight, reduction='mean')
        if self._w:
            ce = ce * self._w
        return ce

class BipartiteMatchingLoss(nn.Module):
    def __init__(self, class_weight=1., l1_weight=5., giou_weight=2.,
                 num_classes=1, eos_coef=0.1):
        super(BipartiteMatchingLoss, self).__init__()
        self._matcher = HungarianMatcher(
            class_weight=class_weight,
            l1_weight=l1_weight,
            giou_weight=giou_weight,
        )
        self._clf = ClassificationLoss(
            weight=class_weight,
            num_classes=num_classes,
            eos_coef=eos_coef
        )
        self._regr = RegressionLoss(l1_weight, giou_weight)

    def forward(self, prediction, targets):
        matching = self._matcher(
            outputs=prediction,
            targets=targets,
        )

        aligned_prediction_boxes = []
        aligned_target_boxes = []
        aligned_prediction_logits = []
        aligned_target_labels = []

        for img_matching, img_targets, pred_logits, pred_boxes in \
                zip(matching, targets, prediction['logits'], prediction['boxes']):
            if len(img_targets) == 0:
                continue
            pred_ids, tgt_ids = img_matching
            for pred_idx, tgt_idx in zip(pred_ids, tgt_ids):
                aligned_prediction_boxes.append(
                    torch.unsqueeze(pred_boxes[pred_idx], 0)
                )
                aligned_target_boxes.append(
                    torch.unsqueeze(img_targets['boxes'][tgt_idx], 0)
                )
                aligned_prediction_logits.append(
                    torch.unsqueeze(pred_logits[pred_idx], 0)
                )
                aligned_target_labels.append(
                    torch.unsqueeze(img_targets['labels'][tgt_idx], 0)
                )
            for i in range(len(pred_logits)):
                if i in pred_ids:
                    continue
                aligned_prediction_logits.append(
                    torch.unsqueeze(pred_logits[i], 0)
                )
                aligned_target_labels.append(
                    torch.tensor([0]).long().to(pred_logits.device)
                )

        aligned_prediction_boxes = torch.cat(aligned_prediction_boxes)
        aligned_target_boxes = torch.cat(aligned_target_boxes)

        regression_loss = self._regr(
            predictions=aligned_prediction_boxes,
            targets=aligned_target_boxes,
        )

        aligned_prediction_logits = torch.cat(aligned_prediction_logits)
        aligned_target_labels = torch.cat(aligned_target_labels)

        classification_loss = self._clf(
            logits=aligned_prediction_logits,
            targets=aligned_target_labels,
        )

        with torch.no_grad():
            target_objects = sum(
                [len(x['labels']) for x in targets]
            )
            pred_objects = torch.argmax(prediction['scores'], dim=-1)
            pred_objects = torch.sum(pred_objects != 0)
            cardinality_error = target_objects - pred_objects

        return {
            'classification': classification_loss,
            'regression': regression_loss,
            'cardinality_error': cardinality_error,
            'loss': classification_loss + regression_loss,
        }
