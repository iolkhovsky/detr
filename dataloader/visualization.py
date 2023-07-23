import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import torch
import torchvision

from util.torch import tensor2numpy


def plot_images_list(images_list, labels_list=None, cols=4, cmap='gray', figsize=(16, 16)):
    sample_size = len(images_list)
    rows = sample_size // cols
    if sample_size % cols:
        rows += 1
    fig, m_axs = plt.subplots(rows, cols, figsize=figsize)
    for idx, (img, c_ax) in enumerate(zip(images_list, m_axs.flatten())):
        c_ax.imshow(img, cmap=cmap)
        label = f'image #{idx}'
        if labels_list is not None:
            label = labels_list[idx]
        c_ax.set_title(f'{label}')
        c_ax.axis('off')


def visualize_boxes(image, boxes=[], labels=[], scores=[], codec=None, width=4): 
    boxes = torch.from_numpy(np.array(boxes))
    image_cwh = torch.from_numpy(
        np.transpose(image, [2, 0, 1])
    )
    boxes_, labels_, colors_ = None, None, None
    vis = image_cwh.clone()

    if len(boxes):
        for obj_idx, (box, label) in enumerate(zip(boxes, labels)):
            label = codec.decode(label)
            color = codec.color(label)
            score = 1.
            if len(scores):
                score = scores[obj_idx]
                label += f": {'{:.2f}'.format(score)}"
            
            obj_width = max(1, int(width * score))

            vis = torchvision.utils.draw_bounding_boxes(
                        image=vis,
                        boxes=torch.stack([box]),
                        labels=[label],
                        width=obj_width,
                        colors=color,
                    )
    return np.transpose(vis.numpy(), [1, 2, 0])


def visualize_batch(imgs_batch, boxes_batch=None, labels_batch=None, scores_batch=None, codec=None, return_images=False, width=2):
    images_bhwc = tensor2numpy(imgs_batch).astype(np.uint8)
    vis_images = []
    for image_idx, image in enumerate(images_bhwc):
        if boxes_batch and labels_batch:
            boxes = tensor2numpy(boxes_batch[image_idx])
            labels = tensor2numpy(labels_batch[image_idx])
            scores = []
            if scores_batch:
                scores = tensor2numpy(scores_batch[image_idx])
            
            image = visualize_boxes(image, boxes, labels, scores, codec=codec, width=width)
        vis_images.append(image)
    if return_images:
        return vis_images
    else:
        plot_images_list(vis_images)


def visualize_pca(data, title, min_limit=3, image_size=(300, 300), dpi=100):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)

    matplotlib.use('Agg')
    width, height = image_size
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)

    x_lim = max(abs(pca_result[:, 0].min()), abs(pca_result[:, 0].max()))
    y_lim = max(abs(pca_result[:, 1].min()), abs(pca_result[:, 1].max()))
    lim = max(x_lim, y_lim, min_limit) * 1.1

    ax.set_xlim([-lim, lim])
    ax.set_xlabel('PC #1')
    ax.set_ylim([-lim, lim])
    ax.set_ylabel('PC #1')
    ax.grid(True)
    ax.set_title(title)

    ax.scatter(pca_result[:, 0], pca_result[:, 1], marker='o', s=100)
    for i, (x, y) in enumerate(pca_result):
        ax.annotate(f'Vector {i}', (x, y), textcoords='offset points', xytext=(5, 5), ha='center')

    fig.canvas.draw()
    graph_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    graph_image = graph_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return graph_image
