import torch
import torch.nn as nn

from model.matcher import HungarianMatcher


class RegressionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        return 0.


class ClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        return 0.

class BipartiteMatchingLoss(nn.Module):
    def __init__(self):
        super(BipartiteMatchingLoss, self).__init__()
        self._matcher = HungarianMatcher()
        self._clf = ClassificationLoss()
        self._regr = RegressionLoss()

    def forward(self, prediction, targets):
        matching = self._matcher(
            outputs=prediction,
            targets=targets,
        )

        aligned_prediction_boxes = []
        aligned_target_boxes = []
        aligned_prediction_scores = []
        aligned_target_labels = []

        for img_matching, img_targets, pred_scores, pred_boxes in \
                zip(matching, targets, prediction['scores'], prediction['boxes']):
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
                aligned_prediction_scores.append(
                    torch.unsqueeze(pred_scores[pred_idx], 0)
                )
                aligned_target_labels.append(
                    torch.unsqueeze(img_targets['labels'][tgt_idx], 0)
                )

        if len(aligned_target_labels) == 0:
            return {
                'classification': -1.,
                'regression': -1.,
                'cardinality_error': 0.,
            }

        aligned_prediction_boxes = torch.cat(aligned_prediction_boxes)
        aligned_target_boxes = torch.cat(aligned_target_boxes)
        aligned_prediction_scores = torch.cat(aligned_prediction_scores)
        aligned_target_labels = torch.cat(aligned_target_labels)

        regression_loss = self._regr(
            predictions=aligned_prediction_boxes,
            targets=aligned_target_boxes,
        )

        classification_loss = self._clf(
            predictions=aligned_prediction_boxes,
            targets=aligned_target_boxes,
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
        }
