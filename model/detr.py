import torch
import torch.nn as nn

from model.layers import (
    DetrPreprocessor, DetrBackbone, LearnablePosEmbeddings2d,
    DetrHead
)


class DETR(nn.Module):
    def __init__(self, resolution=(224, 224), hidden_dim=256, classes=1,
                 attention_heads=8, transformer_layers=6, queries=100):
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
        self._query = nn.Parameter(torch.rand(1, queries, hidden_dim))
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

    def forward(self, x):
        preprocessed, _ = self._preprocessor(x)
        img_features = self._backbone(preprocessed)

        kv = self._position_emb(img_features)
        # q = 

        features = self._transformer(
            src=kv,
            tgt=self._query,
        )

        logits, boxes = self._head(features)
        return logits, boxes
