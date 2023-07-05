# coding=utf-8
import configparser
from glob import glob
import os
import random
import time
from math import ceil
import cv2

from src.util.tools import *
from torch.utils import data
from torch.utils.data import DataLoader


class IrisTest_SCALE(data.Dataset):
    def __init__(self,
                 data_path='../../datasets/CASIA/test/JPEGImages/',
                 input_res=(256,256), # h,w
                 format = '.JPEG'
                 ):
        # set transorformed resolution
        self._input_res = np.array(input_res)

        # Create global index over all specified keys
        self._data_path = data_path
        self._format = format

        self._name_list = [os.path.basename(name).split('.')[0]
                     for name in glob(os.path.join(self._data_path,'*'))]

        assert len(self._name_list) > 0

    def data_normalization_for_test(self, raw_img):
        ih, iw, ic = raw_img.shape

        # Scale image to fit output dimensions (with a little bit of noise)
        if iw >= ih:
            f1 = float(self._input_res[1]) / iw
        else:
            f1 = float(self._input_res[0]) / ih

        raw_img = cv2.resize(raw_img, None, fx=f1, fy=f1, interpolation=cv2.INTER_CUBIC)

        pad_height1 = np.maximum(0, int(self._input_res[0] - raw_img.shape[0]))
        pad_width1 = np.maximum(0, int(self._input_res[1] - raw_img.shape[1]))

        raw_img = cv2.copyMakeBorder(raw_img, 0, pad_height1, 0, pad_width1, cv2.BORDER_CONSTANT,
                                     value=[0, 0, 0]) if pad_height1 + pad_width1 > 0 else raw_img

        # preprocessing for NN
        raw_img = raw_img.astype(np.float32)
        mean = np.array([0.406, 0.456, 0.485])  # 注意这里变了，需要按照bgr的顺序排列
        std = np.array([0.225, 0.224, 0.229])
        raw_img = raw_img / 255.0
        raw_img -= mean
        raw_img /= std
        _raw_img = im_to_torch(raw_img)

        return _raw_img, (pad_height1, pad_width1), 1./f1

    def __getitem__(self, item):
        name = self._name_list[item]
        img_path = os.path.normcase(os.path.join(self._data_path, name + self._format))

        # (H*W*3) 0-255
        raw_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # original shape
        original_shape = raw_img.shape

        _raw_img, _pads, _f_inv = self.data_normalization_for_test(raw_img)
        entry = {
            'name': name,
            'raw_image': _raw_img,
            'pad': _pads,
            'org_shape': original_shape,
            'scale_inv': _f_inv  #实际上测试时圆心和半径进行缩放（*scale_inv），角度不变
        }

        return entry

    def __len__(self):
        return len(self._name_list)



if __name__ == '__main__':
    test_dataset = IrisTest_SCALE(data_path='../../datasets/CASIA/test/JPEGImages/',
                 input_res=(512,512), # h,w
                 format = '.JPEG')

    test_data_loader = DataLoader(test_dataset, 6, shuffle=False, num_workers=4)

    # cpu or gpu?
    device_gpu =  "cuda:7"
    if torch.cuda.is_available() and device_gpu is not None:
        device = torch.device(device_gpu)
    else:
        if not torch.cuda.is_available():
            print("hey man, buy a GPU!")
        device = torch.device("cpu")

# 可以通过tools.crop_image and resize_image恢复原始结果，对于参数仅仅应用缩放即可,见Ellipse.restore_org_size
    for ii, batch in enumerate(test_data_loader):
        raw_image = batch['raw_image'].to(device)
        pad = batch['pad']
        org_shape = batch['org_shape']
        scale_inv = batch['scale_inv']
