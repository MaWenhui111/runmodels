import os
from glob import glob

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from data.labels import labels, id2label
from lib2.helper_pytorch2 import one_hot2dist

from lib2 import args


# img = '1.jpg'
# gt = '1.png'
# INPUT_SHAPE = (640, 640, 1)
#
# im = Image.open('/sdata/wenhui.ma/program/models/EyeSeg-master2/EyeSeg-master/data/Iris-Seg/' + 'train/image/' + img).convert("L")  # 打开图像并转为灰度图
# im = np.array(im, dtype=np.float32)
#
# print(im.shape)
# flat = Image.open('/sdata/wenhui.ma/program/models/EyeSeg-master2/EyeSeg-master/data/Iris-Seg/' + 'train/label/' + gt).convert('L')  # 读取.npy文件
# flat = np.array(flat, dtype=np.uint8)
# flat = (flat == 255).astype(np.uint8)
# print(flat.shape)
# # Pad
# pad_height = np.maximum(0, int(INPUT_SHAPE[0] - im.shape[0]))
# pad_width = np.maximum(0, int(INPUT_SHAPE[1] - im.shape[1]))
# if pad_height + pad_width > 0:
#     im = cv2.copyMakeBorder(im, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT,
#                             value=0)
#     flat = cv2.copyMakeBorder(flat, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT,
#                               value=0)
#
# im = np.expand_dims(im, axis=-1) if INPUT_SHAPE[-1] == 1 else im
# name = img.split('.')[0]  # 图像名字
# im = np.moveaxis(im, -1, 0)  # 维度转换
# print(im.shape, flat.shape)

# target = torch.tensor([[0, 1, 1],[1, 0, 1]])
# print(target)
# Label = (np.arange(2) == target.numpy()[..., None]).astype(np.uint8)
# print(Label)

img = cv2.imread('/sdata/wenhui.ma/program/models/EyeSeg-master2/EyeSeg-master/data/Iris-Seg/test/image/985.jpg', cv2.IMREAD_GRAYSCALE)
# gt = cv2.imread('/sdata/wenhui.ma/program/models/EyeSeg-master2/EyeSeg-master/data/Iris-Seg/test/label/985.png')
flat = Image.open('/sdata/wenhui.ma/program/models/EyeSeg-master2/EyeSeg-master/data/Iris-Seg/test/label/985.png').convert('L')  # 读取.npy文件
flat = np.array(flat, dtype=np.uint8)
flat = (flat == 255).astype(np.uint8)*255
st = np.hstack((img, flat))
cv2.imwrite('985_stack3.png', st)
