import pytest
import torch

from model import HungarianMatcher


def test_matcher():
    targets = [
        {
            'labels': [2, 1],
            'boxes': [
                [100, 100, 40, 40],
                [30, 40, 30, 40],
            ]
        },
        {
            'labels': [1],
            'boxes': [
                [100, 20, 30, 30],
            ]
        },
    ]

    prediction = {
        'scores': [
            [
                [0.1, 0.1, 0.8],
                [0.1, 0.1, 0.8],
                [0.8, 0.1, 0.1],
                [0.3, 0.5, 0.2],
            ],
            [
                [0.9, 0.05, 0.05],
                [0.7, 0.1, 0.2],
                [0.01, 0.98, 0.01],
                [0.05, 0.9, 0.05],
            ],
        ],
        'boxes': [
            [
                [98, 98, 40, 40],
                [100, 110, 40, 42],
                [300, 300, 300, 300],
                [28, 42, 30, 40],
            ],
            [
                [20, 20, 20, 20],
                [30, 30, 30, 30],
                [100, 20, 50, 30],
                [100, 20, 32, 32],
            ],
        ]
    }

    targets = [
        {
            k: torch.tensor(v, dtype=torch.int32 if k == 'labels' else torch.float32)
            for k, v in x.items()
        }
        for x in targets
    ]

    prediction = {
        k: torch.tensor(v, dtype=torch.float32)
        for k, v in prediction.items()
    }

    batch_size, classes, queries = 2, 3, 4
    assert len(targets) == batch_size
    assert prediction['scores'].shape == (batch_size, queries, classes)
    assert prediction['boxes'].shape == (batch_size, queries, 4)

    pred_indices, target_indices = HungarianMatcher()(
        outputs=prediction,
        targets=targets,
    )

