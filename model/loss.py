import torch
import torch.nn as nn

from model.matcher import HungarianMatcher


class RegressionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(prediction, target):
        pass


class ClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(prediction, target):
        pass

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
        print(matching)
