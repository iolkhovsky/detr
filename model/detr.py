import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import (
    DetrPreprocessor, DetrBackbone, LearnablePosEmbeddings2d,
    DetrHead, DetrPostprocessor
)
from model.loss import BipartiteMatchingLoss
from util import box_cxcywh_to_xyxy


class DETR(nn.Module):
    def __init__(self, resolution=(224, 224), hidden_dim=128, classes=1,
                 attention_heads=2, transformer_layers=2, queries=10,
                 class_weight=1., l1_weight=5., giou_weight=2.,
                 **kwargs):
        super(DETR,self).__init__()
        h, w = resolution
        self._preprocessor = DetrPreprocessor(
            target_resolution=resolution,
        )
        self._backbone = DetrBackbone(
            projection_out_chan=hidden_dim,
        )
        self._position_emb = LearnablePosEmbeddings2d(
            height=7,
            width=7,
            hidden_dim=hidden_dim,
        )
        self._query = nn.Parameter(torch.randn(1, queries, hidden_dim))
        self._transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=attention_heads,
            num_encoder_layers=transformer_layers,
            num_decoder_layers=transformer_layers,
            batch_first=True,
        )
        self._head = DetrHead(
            hidden_dim=hidden_dim,
            classes=classes,
        )
        self._postprocessor = DetrPostprocessor(
            height=h,
            width=w,
        )
        self._criterion = BipartiteMatchingLoss(
            class_weight=class_weight,
            l1_weight=l1_weight,
            giou_weight=giou_weight,
        )

    def forward(self, x, targets=None, return_embeddings=False):
        if targets is None:
            self.eval()
        else:
            self.train()

        preprocessed, scales = self._preprocessor(x)
        img_features = self._backbone(preprocessed)
        kv = self._position_emb(img_features)

        b, _, _, _ = img_features.shape
        q = self._query.repeat([b, 1, 1])

        features = self._transformer(
            src=kv,
            tgt=q,
        )
        logits, norm_xywh_boxes = self._head(features)

        norm_xyxy_boxes = box_cxcywh_to_xyxy(norm_xywh_boxes)
        scores = F.softmax(logits, dim=-1)
        xyxy_boxes = self._postprocessor(norm_xyxy_boxes, scales)

        prediction = {
            'scores': scores,
            'boxes': xyxy_boxes,
        }

        if targets is not None:
            prediction.update(
                self._criterion(
                    prediction={
                        'boxes': norm_xyxy_boxes,
                        'logits': logits,
                        'scores': scores,
                    },
                    targets=targets,
                )
            )

        if return_embeddings:
            return prediction, features
        else:
            return prediction
