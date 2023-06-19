import albumentations as A
import numpy as np
import itertools
import torchvision
import torch
from torch.utils.data import DataLoader

from dataloader.voc_labels import VocLabelsCodec


class VocPreprocessor:
    def __init__(self):
        self._pipeline = A.Compose(
            [
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10),
                A.SmallestMaxSize(256, interpolation=1),
                A.RandomCrop(width=256, height=256),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
            ],
            bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3)
        )

    def __call__(self, image, bboxes):
        preprocessed = VocPreprocessor.preprocess_sample((image, bboxes))
        r = self._pipeline(image=preprocessed['image'], bboxes=preprocessed['bboxes'])
        return r['image'], r['bboxes']
    
    @staticmethod
    def preprocess_sample(sample):
        image, annotation = sample
        image = np.array(image)
        annotation = annotation['annotation']
        boxes = []
        for obj in annotation['object']:
            boxes.append([
                float(obj['bndbox']['xmin']),
                float(obj['bndbox']['ymin']),
                float(obj['bndbox']['xmax']),
                float(obj['bndbox']['ymax']),
                obj['name']
            ])
        return {
            'image': image,
            'bboxes': boxes,
        }


def disbatch(boxes, labels, obj_amount):
    offsets = list(itertools.accumulate(obj_amount, initial=0))
    boxes = [boxes[offset:offset + objects_num] for objects_num, offset in zip(obj_amount, offsets)]
    labels = [labels[offset:offset + objects_num] for objects_num, offset in zip(obj_amount, offsets)]
    return boxes, labels


def build_dataloader(subset='train', batch_size=4, shuffle=True, download=False, root="vocdata", target_classes=None):
    codec = VocLabelsCodec(target_classes=target_classes)

    def collate(batch):
        images, bboxes, labels, objects_cnt = [], [], [], []
        for image, img_targets in batch:
            images.append(image)
            img_boxes, img_labels = [], []
            for obj in img_targets:
                box, label = obj[:-1], obj[-1]
                if label in codec:
                    img_boxes.append(box)
                    img_labels.append(codec.encode(label))
            bboxes.extend(img_boxes)
            labels.extend(img_labels)
            objects_cnt.append(len(img_labels))
        return (
            torch.Tensor(images),
            torch.FloatTensor(np.asarray(bboxes)),
            torch.IntTensor(np.asarray(labels)),
            torch.IntTensor(np.asarray(objects_cnt)),
        )

    dataset = torchvision.datasets.VOCDetection(
        root=root,
        year="2012",
        image_set=subset,
        download=download,
        transforms=VocPreprocessor(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
