import numpy as np
import torch
import torch.nn as nn
import torchvision


class DetrPreprocessor(nn.Module):
    def __init__(self, target_resolution=(224, 224)):
        super(DetrPreprocessor, self).__init__()
        self._height, self._width = target_resolution
        self._normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self._dynamic_range = 255.

    def _prepare_single_image(self, tensor_or_array):
        assert len(tensor_or_array.shape) == 3
        if isinstance(tensor_or_array, np.ndarray):
            tensor = torch.from_numpy(tensor_or_array)
        else:
            tensor = tensor_or_array
        if tensor.shape[0] != 3 and tensor.shape[-1] == 3:
            tensor = torch.permute(tensor, (2, 0, 1))
        _, h, w = tensor.shape
        scale = (1., 1.)
        if (h, w) != (self._height, self._width):
            scale = (h / self._height, w / self._width)
            tensor = torchvision.transforms.Resize(
                size=(self._height, self._width),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            )(tensor)
        return tensor.float(), scale

    def forward(self, img_or_batch):
        if isinstance(img_or_batch, list) or len(img_or_batch.shape) == 4:
            tensors_and_scales = [self._prepare_single_image(x) for x in img_or_batch]
            tensors, scales = zip(*tensors_and_scales)
            tensors = torch.stack(tensors)
        else:
            tensor, scale = self._prepare_single_image(img_or_batch)
            tensors, scales = torch.unsqueeze(tensor, 0), [scale]
        b, c, h, w = tensors.shape
        assert c == 3 and h == self._height and w == self._width and b > 0, \
            f"b, h, w, c = {b, h, w, c}"
        tensors /= self._dynamic_range
        normalized_tensors = self._normalize(tensors)
        return normalized_tensors, scales


class DetrBackbone(nn.Module):
    def __init__(self, projection_in_chan=2048, projection_out_chan=256,
                 backbone_type='resnet50', pretrained=True, prune_layers=2):
        super(DetrBackbone,self).__init__()
        cls = getattr(torchvision.models, backbone_type)
        self.backbone = nn.Sequential(
            *list(
                cls(pretrained=pretrained).children()
            )[:-prune_layers]
        )
        self.projection = nn.Conv2d(projection_in_chan, projection_out_chan, 1)

    def forward(self, x):
        return self.projection(self.backbone(x))


class DetrHead(nn.Module):
    def __init__(self, hidden_dim=256, classes=1):
        super(DetrHead,self).__init__()
        self._classes = classes + 1
        self._to_boxes = nn.Linear(hidden_dim, 4)
        self._to_classes = nn.Linear(hidden_dim, self._classes)

    def forward(self, x):
        logits = self._to_classes(x)
        boxes = self._to_boxes(x).sigmoid()
        return logits, boxes
