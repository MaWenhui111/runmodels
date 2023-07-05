import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F


def plot_img_and_mask(img, mask):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()


def get_predictions(output):
    bs, c, h, w = output.size()
    values, indices = output.max(1)
    indices = indices.view(bs, h, w)  # bs x h x w
    return indices


def per_class_mIoU(predictions, targets, info_print=False):
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    unique_labels = np.unique(targets)  # 去掉重复值
    ious = list()
    for index in unique_labels:
        pred_i = predictions == index
        label_i = targets == index
        intersection = np.logical_and(label_i, pred_i)
        union = np.logical_or(label_i, pred_i)
        iou_score = np.sum(intersection) / np.sum(union)
        iou_score = iou_score if np.isfinite(iou_score) else 0.0
        ious.append(iou_score)
    if info_print:
        print("per-class mIOU: ", ious)
    return np.average(ious)


def pad(x1, x2):
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]

    Pad_x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
    return Pad_x1
