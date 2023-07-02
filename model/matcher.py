# mather from https://github.com/facebookresearch/detr/blob/main/models/matcher.py

import torch
from torch import nn
from scipy.optimize import linear_sum_assignment

from util import generalized_box_iou, box_cxcywh_to_xyxy


class HungarianMatcher(nn.Module):
    def __init__(self, class_weight=1., l1_weight=5., giou_weight=2.):
        super().__init__()
        self._w_class = class_weight
        self._w_l1 = l1_weight
        self._w_giou = giou_weight

    @torch.no_grad()
    def forward(self, outputs, targets):
        b, n = outputs['scores'].shape[:2]

        out_prob = outputs['scores'].flatten(0, 1)
        out_bbox = outputs['boxes'].flatten(0, 1)

        target_labels = torch.cat([img_targets['labels'] for img_targets in targets])
        target_bbox = torch.cat([img_targets['boxes'] for img_targets in targets])

        cost_class = -out_prob[:, target_labels]
        cost_bbox = torch.cdist(out_bbox, target_bbox, p=1)
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox),
            box_cxcywh_to_xyxy(target_bbox)
        )

        C = self._w_l1 * cost_bbox + self._w_class * cost_class + self._w_giou * cost_giou
        C = C.view(b, n, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]
