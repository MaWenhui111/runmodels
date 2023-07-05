#!/usr/bin/env python3
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data.labels import labels, trainId2label, id2label
from torch import einsum
from torch.autograd import Variable
from lib2.helper_pytorch2 import flatten, make_one_hot

class BinaryDiceLoss(nn.Module):

    def __init__(self, smooth=1e-7, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "batch size doesn't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        predict = predict.cuda().float()
        target = target.cuda().float()
        

        num = torch.sum(torch.mul(predict, target), dim=1).clamp(min=self.smooth) 
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1).clamp(min=self.smooth) 

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, smooth=1e-7, **kwargs):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                total_loss += dice_loss
       
        return total_loss / (target.shape[1]) 

class GeneralizedDiceLoss(nn.Module):

    def __init__(self, eps=1e-7):
        super(GeneralizedDiceLoss, self).__init__()
        self.eps = eps
        self.norm = nn.Softmax(dim=1)

    def forward(self, ip, target):
        Label = (np.arange(2) == target.cpu().numpy()[..., None]).astype(np.uint8)
        target = torch.from_numpy(np.rollaxis(Label, 3,start=1)).cuda()
        assert ip.shape == target.shape
        ip = self.norm(ip)
        ip = torch.flatten(ip, start_dim=2, end_dim=-1).cuda().to(torch.float32)
        target = torch.flatten(target, start_dim=2, end_dim=-1).cuda().to(torch.float32)
        numerator = ip*target
        denominator = ip + target
        class_weights = 1./(torch.sum(target, dim=2)**2).clamp(min=self.eps)
        A = class_weights*torch.sum(numerator, dim=2)
        B = class_weights*torch.sum(denominator, dim=2)
        dice_metric = 2.*torch.sum(A, dim=1)/torch.sum(B, dim=1)
        return torch.mean(1. - dice_metric.clamp(min=self.eps))

class SurfaceLoss(nn.Module):

    def __init__(self, epsilon=1e-7, softmax=True):
        super(SurfaceLoss, self).__init__()
        self.weight_map = []
    def forward(self, x, distmap):
        x = torch.softmax(x, dim=1)
        self.weight_map = distmap
        score = x.flatten(start_dim=2)*distmap.flatten(start_dim=2)
        score = torch.mean(score, dim=2)
        score = torch.mean(score, dim=1) 
        return score

class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None):
        super(CrossEntropyLoss2d,self).__init__()
        self.loss = nn.NLLLoss(weight)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs,dim=1), targets)

class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, outputs):
        plog = F.log_softmax(outputs, dim=1)
        p = F.softmax(outputs, dim=1)
        return - p.mul(plog)