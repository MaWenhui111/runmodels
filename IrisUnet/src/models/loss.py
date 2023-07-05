from __future__ import division
import torch
import torch.nn
import numpy as np
from torch.nn import functional as F

class Mask_Loss():
    def __init__(self, seg_weight=1):
        self.seg_weight = seg_weight

    def __call__(self, pred_mask, gt_mask_image):
        gt_mask_image = gt_mask_image.squeeze(1).long()
        mask_loss = 0

        # dirty label check
        if not ((gt_mask_image >= 0).all and(gt_mask_image < 2).all()):
            gt_mask_image = torch.where(gt_mask_image>1, 1, gt_mask_image)
            gt_mask_image = torch.where(gt_mask_image<0, 0, gt_mask_image)

        for seg in pred_mask:
            mask_loss += self.seg_weight * torch.nn.CrossEntropyLoss(size_average=True)(seg, gt_mask_image)
        loss = mask_loss


        return loss, mask_loss



'''
class Mask_edge_ce_Loss():
    def __init__(self, seg_weight=1, edge_weight=1):
        self.seg_weight = seg_weight
        self.edge_weight = edge_weight

    def __call__(self, pred_mask, pred_edge, gt_mask_image, gt_edge_image):
        gt_mask_image = gt_mask_image.squeeze(1).long()
        gt_edge_image = gt_edge_image.squeeze(1).long()

        mask_loss = 0
        edge_loss = 0

#  如下这步移到IrisNet前面
        # if not isinstance(pred_mask, list):
        #     pred_mask = [pred_mask]
        # if not isinstance(pred_edge, list):
        #     pred_edge = [pred_edge]

        for seg in pred_mask:
            mask_loss += self.seg_weight * torch.nn.CrossEntropyLoss(size_average=True)(seg, gt_mask_image)

        for edge in pred_edge:
            edge_loss += self.edge_weight * torch.nn.CrossEntropyLoss(size_average=True)(edge, gt_edge_image)

        loss = mask_loss + edge_loss

        return loss, mask_loss, edge_loss
'''

# https://github.com/scaelles/DEXTR-PyTorch/blob/master/layers/loss.py
def class_balanced_cross_entropy_loss(output, label, size_average=True, batch_average=True, void_pixels=None):
    """Define the class balanced cross entropy loss to train the network
    Args:
    output: Output of the network N*c*H*W
    label: Ground truth label  N*1*H*W
    size_average: return per-element (pixel) average loss
    batch_average: return per-batch average loss
    void_pixels: pixels to ignore from the loss
    Returns:
    Tensor that evaluates the loss
    """
    num_classes = output.size(1)
    multi_labels = []
    for i in range(num_classes):
        multi_labels.append(1.0 * (label == label.new_full(label.size(), i+1)).float())
    multi_labels = torch.cat(multi_labels, 1)
    assert(output.size() == multi_labels.size())

    labels = torch.ge(multi_labels, 0.5).float()

    num_labels_pos = torch.sum(labels)
    num_labels_neg = torch.sum(1.0 - labels)
    num_total = num_labels_pos + num_labels_neg

    output_gt_zero = torch.ge(output, 0).float()
    loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
        1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))

    loss_pos_pix = -torch.mul(labels, loss_val)
    loss_neg_pix = -torch.mul(1.0 - labels, loss_val)

    if void_pixels is not None:
        w_void = torch.le(void_pixels, 0.5).float()
        loss_pos_pix = torch.mul(w_void, loss_pos_pix)
        loss_neg_pix = torch.mul(w_void, loss_neg_pix)
        num_total = num_total - torch.ge(void_pixels, 0.5).float().sum()

    loss_pos = torch.sum(loss_pos_pix)
    loss_neg = torch.sum(loss_neg_pix)

    if num_labels_neg == 0:
        final_loss = loss_pos
    elif num_labels_pos == 0:
        final_loss = loss_neg
    else:
        final_loss = num_labels_neg / num_total * loss_pos + num_labels_pos / num_total * loss_neg

    if size_average:
        final_loss /= np.prod(multi_labels.size()).astype(np.float)
    elif batch_average:
        final_loss /= multi_labels.size(0)

    return final_loss

class Mask_edge_bce_Loss():
    def __init__(self, seg_weight=1, edge_weight=1):
        self.seg_weight = seg_weight
        self.edge_weight = edge_weight

    def __call__(self, pred_mask, pred_edge, gt_mask_image, gt_edge_image):
        gt_mask_image = gt_mask_image.squeeze(1).long()

        mask_loss = 0
        edge_loss = 0

        for seg in pred_mask:
            mask_loss += self.seg_weight * torch.nn.CrossEntropyLoss(size_average=True)(seg, gt_mask_image)

        for edge in pred_edge:
            edge_loss += self.edge_weight * class_balanced_cross_entropy_loss(edge, gt_edge_image, size_average=True, batch_average=False, void_pixels=None)

        loss = mask_loss + edge_loss

        return loss, mask_loss, edge_loss