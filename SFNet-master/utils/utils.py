import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch
from torchvision import transforms
import random
from PIL import Image


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
    predictions = predictions.cpu().numpy() if type(predictions) != np.ndarray else predictions
    targets = targets.cpu().numpy() if type(targets) != np.ndarray else targets
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


def mask_to_image(mask: np.ndarray, mask_values):  # mask映射
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v
    return out
    # return Image.fromarray(out)


def pad(x1, x2):
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]

    Pad_x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
    return Pad_x1


def data_augmentation(image, mask):
    # Flip
    flip_methods = [transforms.functional.hflip,  # 水平翻转
                    transforms.functional.vflip,  # 垂直翻转
                    lambda x: transforms.functional.rotate(x, 90)  # 水平和垂直翻转
                    ]
    if random.random() < 1 / 3:
        flip_method = random.choice(flip_methods)
        image = flip_method(image)
        mask = flip_method(mask)

    # Gaussian Blur
    if random.random() < 1 / 3:
        image = transforms.functional.gaussian_blur(image, kernel_size=3, sigma=1)

    # Random Cropping
    if random.random() < 1 / 3:
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(48, 64))
        image = transforms.functional.crop(image, i, j, h, w)
        mask = transforms.functional.crop(mask, i, j, h, w)

    return image, mask
