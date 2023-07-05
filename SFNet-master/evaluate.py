import logging
import numpy as np
import cv2
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
import gc
from utils.utils import per_class_mIoU, get_predictions, mask_to_image


@torch.inference_mode()
def evaluate(net, dataloader, device, criterion, mask_values, args):
    net.eval()
    num_val_batches = len(dataloader)
    # dice_score = 0
    ious = list()
    losses = list()

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=args.amp):
        with torch.no_grad():
            for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
                sclera_img, Iris_img, pupil_img, sclera_mask, Iris_mask, pupil_mask, mask_true, name = batch

                # move images and labels to correct device and type
                sclera_img = sclera_img.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                Iris_img = Iris_img.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                pupil_img = pupil_img.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                mask_true = mask_true.to(device=device, dtype=torch.long)

                # predict the mask
                mask_pred, mask_pred_sclera, mask_pred_Iris, mask_pred_pupil = net(sclera_img, Iris_img, pupil_img)
                mask_pred = mask_pred.to(device=device)
                loss = criterion(mask_pred, mask_true)
                losses.append(loss.detach().item())
                mask_pred = get_predictions(mask_pred)  # 将小数变成0~n_classes的整数,并将mask_pred变成1个通道

                iou = per_class_mIoU(mask_pred, mask_true)  # mask_true为1个通道
                ious.append(iou)

                if args.save:
                    assert args.batch_size == 1
                    mask_pred = mask_pred.squeeze(0)
                    mask_true = mask_true.squeeze(0)
                    mask_pred = mask_pred.cpu().numpy()
                    mask_true = mask_true.cpu().numpy()
                    result2 = mask_to_image(mask_pred, mask_values)
                    result1 = mask_to_image(mask_true, mask_values)
                    TrueAndPred = np.hstack((result1, result2))
                    Path(args.output + 'true_pred/').mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(args.output + 'true_pred/' + name[0] + '.png', result2)


                # if net.n_classes == 1:
                #     assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                #     mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                #     # compute the Dice score
                #     dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                # else:
                #     assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes]'
                #     # convert to one-hot format
                #     mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                #     mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                #     # compute the Dice score, ignoring background
                #     dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
    gc.collect()
    # net.train()
    # return dice_score / max(num_val_batches, 1)
    return np.average(ious), np.average(losses)


