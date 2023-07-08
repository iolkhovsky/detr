import argparse
import cv2
import os
import time
import torch

from dataloader.visualization import visualize_batch
from pl.module import DetrModule
from dataloader.voc_labels import VocLabelsCodec


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',
                        default=os.path.join('checkpoints', 'epoch-0016-loss-0.000000-acc-0.000000.ckpt'),
                        help='Path to DETR checkpoint')
    parser.add_argument('--threshold',
                        default=0.05,
                        help='Confidence threshold')
    args = parser.parse_args()
    return args


def compile_model(checkpoint_path, device=None):
    model = DetrModule().load_from_checkpoint(checkpoint_path, map_location=torch.device('cpu'))
    # if device: 
    #     model = model.to(device)
    return model


def img2tensor(image, res, device=None):
    tensor = torch.tensor([cv2.resize(image, res)])
    if device:
       tensor = tensor.to(device)
    return tensor


def visualize(img, pred, labels_codec, threshold=0.1):
    filtered_boxes, filtered_scores, filtered_labels = [], [], []
    for img_idx in range(1):
        mask = pred['scores'][img_idx][:, 1] > threshold
        filtered_boxes.append(
            pred['boxes'][img_idx][mask]
        )
        filtered_scores.append(
            pred['scores'][img_idx][mask][:, 1]
        )
        filtered_labels.append(
            torch.tensor([1] * len(mask))[mask]
        )  

    vis_pred_imgs = visualize_batch(
        imgs_batch=img,
        boxes_batch=filtered_boxes,
        labels_batch=filtered_labels,
        scores_batch=filtered_scores,
        codec=labels_codec,
        return_images=True
    )
    
    return vis_pred_imgs[0]


class Profiler:
    def __init__(self, hint):
        assert isinstance(hint, str)
        self._hint = hint
        self._start = None

    def __enter__(self):
        self._start = time.time()

    def __exit__(self, type, value, traceback):
        duration = time.time() - self._start
        duration_s = '{:.3f}'.format(duration)
        fps =  '{:.2f}'.format(1. / (duration + 1e-4))
        print(f'{self._hint} duration (sec): {duration_s}\tFPS={fps}')


def run(args):
    model = compile_model(args.checkpoint, 'cpu')

    video_source = str(0)
    if video_source.isdigit():
        video_source = int(video_source)
    resolution = (448, 448)
    threshold = args.threshold
    labels_codec = VocLabelsCodec(['person'])
    cap = cv2.VideoCapture(video_source)

    profiler = Profiler('Inference')

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        img_tensor = img2tensor(frame, resolution, 'cpu')
        with profiler:
            detections = model.model(img_tensor)
        vis_img = visualize(img_tensor, detections, labels_codec, threshold)
        cv2.imshow('Prediction', vis_img)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run(parse_cmd_args())
