import numpy as np
import torch
import gc
from lib2.command_line import command_line_args
from data.labels import trainId2label, id2label
from scipy.ndimage import distance_transform_edt as distance  # 前景像素与最近的背景像素之间的距离

args = command_line_args().parse_args()


def one_hot2dist(posmask):
    assert len(posmask.shape) == 2
    h, w = posmask.shape
    res = np.zeros_like(posmask)
    posmask = posmask.astype(np.bool)
    mxDist = np.sqrt((h - 1) ** 2 + (w - 1) ** 2)
    if posmask.any():
        negmask = ~posmask
        res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res / mxDist


def get_predictions(output):
    bs, c, h, w = output.size()
    values, indices = output.max(1)
    indices = indices.view(bs, h, w)  # bs x h x w
    return indices


def pred2ones(predict, mask_shape):
    tmp = predict.cpu().numpy()
    mask = np.zeros(mask_shape, dtype=np.uint8)
    for img_idx in range(len(mask)):
        for k, v in id2label.items():
            mask[img_idx, v.trainId, :, :] += (tmp[img_idx] == k).astype(np.uint8)
    return torch.from_numpy(mask).cuda().long()


def per_class_mIoU(predictions, targets, info=False):
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
    if info:
        print("per-class mIOU: ", ious)
    return np.average(ious)


def pp_class_miou(classes):
    form = '\t{:20s}\t|\t{:10s}'
    print()
    print(form.format('Class Name', 'mIoU'))
    print(''.join(['-' for i in range(60)]))
    for class_id, iou in classes.items():
        if class_id in id2label.keys():
            string_iou = '{:.4f}'.format(np.mean(iou))
            print(form.format(str(id2label[class_id].name), string_iou))
    print()
    print()


def torch_validate(val_data, model, device, cce_loss, dice_loss, surface_loss):
    ious = list()
    losses = list()
    with torch.no_grad():
        for batch_num, batch in enumerate(val_data):
            input_image, ground_truth, one_hot, spatial_gt, distMap, name = batch
            data_in = input_image.to(device)
            output = model(data_in)
            cce = cce_loss(output.to(device), ground_truth.to(device).long()) * (
                        torch.from_numpy(np.ones(spatial_gt.shape)).to(torch.float32).to(device) + (spatial_gt).to(
                    torch.float32).to(device))
            loss = torch.mean(dice_loss(output.to(device), ground_truth.to(device).long()))
            loss = loss + torch.mean(cce)
            predict = get_predictions(output.to(device))
            iou = per_class_mIoU(predict, ground_truth)
            ious.append(iou)
            losses.append(loss.detach().item())
    avg_acc, avg_loss = np.average(ious), np.average(losses)
    gc.collect()
    return avg_acc, avg_loss


def labelid_to_color(pred):
    color_img = np.zeros((args.INPUT_SHAPE[0], args.INPUT_SHAPE[1], 3), dtype=np.uint8)
    for k, v in trainId2label.items():
        color_img[:, :, 0] += ((pred == k).astype(np.uint8) * v.color[0])
        color_img[:, :, 1] += ((pred == k).astype(np.uint8) * v.color[1])
        color_img[:, :, 2] += ((pred == k).astype(np.uint8) * v.color[2])
    return color_img


def encode_test(pred):
    gray_scale = np.zeros((args.INPUT_SHAPE[0], args.INPUT_SHAPE[1]), dtype=np.uint8)
    for k, v in trainId2label.items():
        gray_scale[:, :] += (pred == k).astype(np.uint8) * v.id
    return gray_scale


def flatten(tensor):
    C = tensor.size(1)
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    transposed = tensor.permute(axis_order).contiguous()
    return transposed.view(C, -1)


def make_one_hot(input, num_classes):
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)
    return result
