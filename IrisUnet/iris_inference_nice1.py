#!/usr/bin/env python
# coding: utf-8

import os
import time
import json
from collections import OrderedDict
import importlib
import logging
import argparse
import numpy as np
import random
import scipy.misc
import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader
import torch.backends.cudnn
from tqdm import tqdm

import sys
sys.path.append('/sdata/wenhui.ma/program/IrisUnet/src/')

import src.datasources as datasources
import src.models as models
from config_nice1 import *
from src.util.osutils import mkdir_p, isfile, isdir, join
from src.util.tools import *

torch.backends.cudnn.benchmark = True

logging.basicConfig(
    format='[%(asctime)s %(name)s %(levelname)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)

def save_seg_pupil_iris(input, pad, org_shape, save_name):
    seg_img = to_numpy(torch.squeeze(input))
    org_seg_img = crop_image(seg_img, pad)
    org_seg_img = resize_image(org_seg_img, org_shape[0], org_shape[1])
#    scipy.misc.imsave(save_name, org_seg_img)
    imageio.imsave(save_name, org_seg_img)


def inference(input_path, model_path, output_path, device, format='.JPEG'):
    if not isfile(model_path):
        logger.info("=> no checkpoint found at '{}'".format(model_path))

    logger.info("=> loading checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path)
    cfg = checkpoint['cfg']
    new_config = {'isTrain': False,
                  'model_home_path':{'resnet18': None,
                                    'vgg16': None
                                     }
        }
    cfg.parse(new_config)

    # 设立输出路径
    save_seg_path = join(output_path, 'mask')
    save_iris_path = join(output_path, 'iris')

    mkdir_p(save_seg_path)
    mkdir_p(save_iris_path)


    # create model
    logger.info("==> creating model '{}'".format(cfg.model_arch))
    model = models.__dict__[cfg.model_arch](cfg, logger)
    model.load_networks(checkpoint)
    model.to(device)
    model.print_networks(False)

    test_dataset = datasources.IrisTest_SCALE(data_path=input_path, input_res=cfg.input_res,  format=format)

    logger.info('Totally %d images' % (len(test_dataset)))
    start = time.time()

    # Start!
    logger.info("Start testing!\n")
    model.eval()
    with torch.no_grad():
        for input in tqdm(test_dataset, ncols=75, ascii=True):
            raw_image = input['raw_image']
            entry = {
                'raw_image': raw_image.unsqueeze(0)
            }
            name = input['name']
            org_shape = input['org_shape']
            pad = input['pad']
            scale_inv = input['scale_inv']
            model.set_input(entry, device)

            # 载入org_image
            # org_img = restore_img(im_to_numpy(raw_image))
            # org_raw_img = crop_image(org_img, pad)
            # org_raw_img = resize_image(org_raw_img, org_shape[0], org_shape[1])

            # model
            model.forward()
            org_seg = model.seg[0]
            org_iris = 1.0 * (model.seg[0] == model.seg[0].new_ones(model.seg[0].size())).float()

            save_seg_pupil_iris(org_seg, pad, org_shape, join(save_seg_path, name + '.png'))
            save_seg_pupil_iris(org_iris, pad, org_shape, join(save_iris_path, name + '.png'))

    end = time.time()
    avg_time = (end - start) / len(test_dataset)
    logger.info("average time is %f seconds" % avg_time)


if __name__ == '__main__':
    input_path = '/sdata/wenhui.ma/program/IrisUnet/data/forseg/test/'
    model_path = '/sdata/wenhui.ma/program/IrisUnet/checkpoints/Iris_NDCLD15_i448x448_0314_223506_UNet/models/best_UNet_NDCLD15.pth'
    output_path = './checkpoints/result/'

    if not isdir(output_path):
        mkdir_p(output_path)

    gpu_device = "cuda:0" # None if set cpu

    if torch.cuda.is_available() and gpu_device is not None:
        device = torch.device(gpu_device)
    else:
        if not torch.cuda.is_available():
            print("hey man, buy a GPU!")
        device = torch.device("cpu")

    format = '.tiff'

    inference(input_path, model_path, output_path, device, format)
