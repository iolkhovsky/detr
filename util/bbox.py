# utils from https://github.com/facebookresearch/detr/blob/main/util/box_ops.py

import torch
from torchvision.ops.boxes import box_area


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1] 

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union



def generalized_box_iou(x0y0x1y1_boxes1, x0y0x1y1_boxes2):
    assert (x0y0x1y1_boxes1[:, 2:] >= x0y0x1y1_boxes1[:, :2]).all()
    assert (x0y0x1y1_boxes2[:, 2:] >= x0y0x1y1_boxes2[:, :2]).all()
    iou, union = box_iou(x0y0x1y1_boxes1, x0y0x1y1_boxes2)

    lt = torch.min(x0y0x1y1_boxes1[:, None, :2], x0y0x1y1_boxes2[:, :2])
    rb = torch.max(x0y0x1y1_boxes1[:, None, 2:], x0y0x1y1_boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def denormalize_boxes(normalized_xyxy, imgs_shapes):
    result = []
    for img_boxes, img_shape in zip(normalized_xyxy, imgs_shapes):
        if len(img_boxes):
            h, w, _ = img_shape
            img_boxes = img_boxes * torch.tensor([w, h] * 2)
        result.append(img_boxes)
    return result
