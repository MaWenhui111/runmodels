import os

import cv2
import numpy as np
import torch
from PIL import Image

from data.labels import labels, id2label
from lib2.helper_pytorch2 import one_hot2dist

from lib2 import args
import pandas as pd


class Loader:
    """this will be a helper class object 'Loader' which will do the actual
    processing of the file IO
 
    Notes: You will not actually use this class object directly
 
    Argurments:
 
        utils: this is the named dict of all the args parser information

    """

    def __init__(self, utils : dict, test : bool, mode : str):
        """
            This constructor will be loading in the pickled label data for binary/onehot masking
        """
        self.utils = utils
        self.labels = labels
        self.mapping = id2label
        self.isTrain = not test
        self.mode = mode

    def __call__(self, img : str, gt : str):
        """The __call__ method will be handling any direct call to this class object
        this will take in two paths: one to the image, and one to the ground truth.

        Note: This returns the numpy data, which the TorchData class will later convert to torch.Tensor

        """
        if self.isTrain:
            if self.mode == 'full':
                im = Image.open(self.utils.ROOT_FOLDER+img).convert("L")  # 打开图像并转为灰度图
                im = np.array(im, dtype = np.float32)
                im = np.expand_dims(im, axis=-1) if self.utils.INPUT_SHAPE[-1] == 1 else im
                name = img.split('.png')[0].split('/')[-1]  # 取图像名字
                im = np.moveaxis(im, -1, 0)  # 维度转换
                flat = np.load(self.utils.ROOT_FOLDER+gt)  # 读取.npy文件
                flat = np.array(flat, dtype=np.uint8)
                mask = np.zeros((flat.shape[0], flat.shape[1], self.utils.NUM_CLASSES), dtype=np.uint8)
                tmp = flat.copy()
                for k, v in self.mapping.items():
                    mask[:,:,v.trainId] += (tmp == k).astype(np.uint8)  # 将每种label分开成不同的通道
                distMap = list()
                for i in range(0, 4):
                    distMap.append(one_hot2dist(np.array(flat)==i))  # ???
                distMap = np.stack(distMap, 0)
                spatialWeights = cv2.Canny(np.array(flat, dtype=np.uint8),0,4)/255
                spatialWeights = cv2.dilate(spatialWeights,(3,3),iterations = 1)
                return im, flat, mask, spatialWeights, np.float32(distMap), name
            elif self.mode == 'semi':
                im = Image.open(self.utils.ROOT_FOLDER+img).convert("L")
                im = np.array(im, dtype = np.float32)
                im = np.expand_dims(im, axis=-1) if self.utils.INPUT_SHAPE[-1] == 1 else im
                im = np.moveaxis(im, -1, 0)
                name = img.split('.png')[0].split('/')[-1]
                if gt is None:
                    return im, [], [], [], [], name
                else:
                    flat = np.load(self.utils.ROOT_FOLDER+gt)
                    flat = np.array(flat, dtype=np.uint8)
                    mask = np.zeros((flat.shape[0], flat.shape[1], self.utils.NUM_CLASSES), dtype=np.uint8)
                    tmp = flat.copy()
                    for k, v in self.mapping.items():
                        mask[:,:,v.trainId] += (tmp == k).astype(np.uint8)  # ????
                    distMap = list()
                    for i in range(0, 4):
                        distMap.append(one_hot2dist(np.array(flat)==i))
                    distMap = np.stack(distMap, 0)
                    spatialWeights = cv2.Canny(np.array(flat, dtype=np.uint8),0,4)/255
                    spatialWeights = cv2.dilate(spatialWeights,(3,3),iterations = 1)
                    return im, flat, mask, spatialWeights, np.float32(distMap), name
        else:
            im = Image.open(self.utils.ROOT_FOLDER+img).convert("L")  # 以灰度图形式打开图像
            im = im.resize((self.utils.INPUT_SHAPE[1], self.utils.INPUT_SHAPE[0]))  # 图像由HxW转换为WxH
            im = np.array(im, dtype = np.float16)  # 将图像转为数组形式
            im = np.expand_dims(im, axis=-1) if self.utils.INPUT_SHAPE[-1] == 1 else im
            im = np.moveaxis(im, -1, 0)  # 将最后一个维度移到第0维，如果INPUT_SHAPE[-1] != 1，就是将WxH转为HxW
            im = np.array(im, dtype=np.float16)
            name = img.split('.png')[0].split('/')[-1]  # 获取图像名字
            return im, [], [], [], [], name

class TorchData(torch.utils.data.Dataset):
    """the dataset object will process text line data mapped as text file x  and text file y or data sample file and label text file
 
    needs to be in string format data that maps to an image and a label that is in the format of a numpy pickled object

    Notes: Please see the if condition for running this as main for an example of how to run this module
 
    Argurments:
 
        x: this will be the text file path/to/file_name.txt that will be a path per line/sample
        y: this will be the text file path/to/file_name.txt that will be a path per line/label
        utils: his is the named dict of all the args parser information
    
    To Use:
 
         from lib import TorchData
         from torch.utils.data import DataLoader
         dataset = TorchData(x = 'relative/path/to/train_data_input.txt', y = 'relative/path/to/train_labels.txt', utils = kargs)
         training = DataLoader( train_data
        ...              , batch_size = args.BATCH_SIZE
        ...              , shuffle=True
        ...              , num_workers = args.WORKERS
        ...              )
         for i, batch in enumerate(training):
        ...     print(batch) # will be a tuple (number of sets per iteration, train image, train label) 
    """
    def __init__(self, x : str, y : str, utils : dict, test = False, mode = 'full'):
        """
            The constructor will parse the given files and strip any new lines.

            additionally, this constructor will init a Loader class object which will
            be utilized for the actual File IO portion
        """
        assert os.path.isfile(x), 'Something is wrong with this path: '+x
        if mode == 'full':
            assert os.path.isfile(y), 'Something is wrong with this path: '+y
            self.images = [s.strip() for s in open(x,'r').readlines()]
            self.labels = [s.strip() for s in open(y,'r').readlines()]
            self.utils = utils
            self.loader = Loader(self.utils, test, mode)
        elif mode == 'semi':
            self.images = [s.strip() for s in open(x,'r').readlines()]  # 删除s开头和结尾处的空白
            if y is None:
                self.labels = [None] * len(self.images)
            else:
                assert os.path.isfile(y), 'Something is wrong with this path: '+y
                self.labels = [s.strip() for s in open(y,'r').readlines()]
            self.utils = utils
            self.loader = Loader(self.utils, test, mode)

    def __len__(self) -> int:
        """ an override of length """
        return len(self.images)

    def __getitem__(self, index : int):
        """an override of the getitem from torch Dataset"""
        im, flat, one_hot, spatialWeights, distMap, name = self.loader(self.images[index],self.labels[index])
        return im, flat, one_hot, spatialWeights, distMap, name

if __name__ == '__main__':
    # pd.set_option('display.max_columns', None)  # 显示所有列
    # pd.set_option('display.max_rows', None)  # 显示所有行
    # loader = Loader(args, False, 'full')
    # loader('image/2.jpg', 'label/2.png')
    img = 'image/2.jpg'
    gt = 'label/2.png'
    im = Image.open(args.ROOT_FOLDER + img).convert("L")
    im = np.array(im, dtype=np.float32)
    print(im.shape)
    im = np.expand_dims(im, axis=-1) if args.INPUT_SHAPE[-1] == 1 else im
    print(im.shape)
    name = img.split('.jpg')[0].split('/')[-1]  # 取图像名字
    print(name)
    im = np.moveaxis(im, -1, 0)  # 维度转换，将最后一维挪到第零维
    print(im.shape)
    flat = Image.open(args.ROOT_FOLDER + gt).convert('L')  # np.load读取.npy文件
    flat = np.array(flat, dtype=np.uint8)
    flat = (flat == 255).astype(np.uint8)
    print((flat == 255).any())
    mask = np.zeros((flat.shape[0], flat.shape[1], args.NUM_CLASSES), dtype=np.uint8)
    tmp = flat.copy()
    for k, v in id2label.items():
        mask[:, :, v.trainId] += (tmp == k).astype(np.uint8)  # ????
    # distMap = list()
    # for i in range(0, 4):
    #      distMap.append(one_hot2dist(np.array(flat) == i))
    # distMap = np.stack(distMap, 0)
    # spatialWeights = cv2.Canny(np.array(flat, dtype=np.uint8), 0, 4) / 255
    # spatialWeights = cv2.dilate(spatialWeights, (3, 3), iterations=1)
    #  im, flat, mask, spatialWeights, np.float32(distMap), name