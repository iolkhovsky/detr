# https://www.w3schools.com/tags/ref_colornames.asp


class VocLabelsCodec:
    def __init__(self, target_classes=None):
        self._names = [
            'person', 'bottle', 'chair', 'diningtable',
            'pottedplant', 'tvmonitor', 'sofa', 'aeroplane',
            'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
            'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
        ]
        if target_classes is not None:
            self._names = [x for x in self._names if x in target_classes]
        self._names = ['background'] + self._names

        self._name2id = dict(zip(self._names, list(range(len(self._names)))))
        self._id2name = {v: k for k, v in self._name2id.items()}

        self._colors = [
            'red', 'aqua', 'brown', 'SandyBrown', 'green', 'blue',
            'purple', 'LightSlateGray', 'SkyBlue', 'orange', 'lime',
            'Olive', 'pink', 'DarkGrey', 'black', 'salmon', 'GreenYellow',
            'yellow', 'Turquoise', 'white'
        ]
        self._colors = ['pink'] + self._colors

    def encode(self, name):
        return self._name2id[name]

    def decode(self, class_idx):
        return self._id2name[class_idx]

    def color(self, idx):
        if isinstance(idx, str):
            idx = self.encode(idx)
        return self._colors[idx]

    def __len__(self):
        return len(self._names)

    def __contains__(self, x):
        return x in self._names
