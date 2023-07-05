import os
import cv2
import torch
import random
import numpy as np
from glob import glob
from torch.utils import data
from torch.utils.data import DataLoader
import sys
from src.datasources.dirty_label_check import check_signal
#sys.path.append(r'D:\Users\userLittleWatermelon\codes\CE_Net_example\CE_Net_example')
sys.path.append('/sdata/wenhui.ma/program/IrisUnet/src/')


from src.util.tools import im_to_torch, im_to_numpy
from torch.utils.data.dataloader import default_collate  # 导入默认的拼接方式

def my_collate_fn(batch):
    # 过滤为None的数据
    batch = list(filter(lambda x: x['raw_image'] is not None, batch))
    return default_collate(batch)  # 用默认方式拼接过滤后的batch数据


# This loader is only for train set and validate set
class Iris(data.Dataset):
    def __init__(self,
                 mode='train',  # or val
                 #data_path='/data1/caiyong.wang/data/IrisLocSeg/',
                 data_path= '/sdata/wenhui.ma/program/IrisUnet/data/Iris-Seg/',
                 input_res=(448, 448),  # h*w  512,512
                 #img_format=['.JPEG', '.png']  # img_format, seg_format
                 # img_format=['.tiff', '.png']
                 img_format=['.jpg', '.png']
                 ):
        # set transform resolution
        self._input_res = np.array(input_res)
        assert (self._input_res % 32 == 0).all()
        self._output_res = self._input_res
        self._mode = mode
        self._img_format = img_format
        assert mode in ["train", "val"]

        # Create global index over all specified keys
        self._root_path = data_path
        self._data_path = os.path.join(self._root_path, 'train')

        name_list = [os.path.basename(name).split('.')[0]
                    for name in glob(os.path.join(self._data_path, 'image', '*'))]

        # name_list = [name.split('.')[0]
        #             for name in os.listdir(os.path.join(self._data_path, 'image'))]


        # divide dataset for train and validate
        random.shuffle(name_list)

        assert len(name_list) > 0
        if self._mode == 'train':
            self._name_list = name_list[:int(0.8 * len(name_list))]

        else:
            self._name_list = name_list[int(0.2 * len(name_list)):]


    def rotate(self, src, angle, flags=cv2.INTER_CUBIC, border_mode=cv2.BORDER_CONSTANT, border_value=0):
        # https://blog.csdn.net/a352611/article/details/51418178
        # https://www.cnblogs.com/Anita9002/p/8033101.html
        # https://opencvpython.blogspot.com/2012/06/contours-2-brotherhood.html
        # get rotation matrix for rotating the image around its center
        height, width = src.shape[:2]
        center = (width / 2.0, height / 2.0)
        rot = cv2.getRotationMatrix2D(center, angle, 1.0)

        # determine bounding rectangle
        boxpoints = cv2.boxPoints((center, (width, height), angle))
        bbox = cv2.boundingRect(boxpoints)  # x,y,width,height

        # adjust transformation matrix
        rot[0, 2] += bbox[2] / 2.0 - center[0]
        rot[1, 2] += bbox[3] / 2.0 - center[1]
        return cv2.warpAffine(src, rot, (bbox[2], bbox[3]), flags=flags, borderMode=border_mode, borderValue=border_value)

    #def data_transform(self, raw_img, mask_img, edge_img, enhance=0.5, randomdict=None):
    def data_transform(self, raw_img, mask_img, enhance=0.5, randomdict=None):
        randomdict = {
            'enhance': enhance,
            'resize': np.random.choice(np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])),
            'blur_type': np.random.randint(5),
            'blur_param': 1 + 2 * np.random.randint(3),
            'translate_tw': round(np.random.uniform(-30, 30)),
            'translate_th': round(np.random.uniform(-30, 30)),
            'p_flip': np.random.rand(),
            'rotate_angle': round(np.random.uniform(-60, 60)),
        } if randomdict is None else randomdict

        # Resize
        f = randomdict['resize']
        raw_img = cv2.resize(raw_img, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        mask_img = cv2.resize(mask_img, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
        # check_signal(mask_img, 'after resize')

        # Blur
        blur_type = randomdict['blur_type']
        blur_param = randomdict['blur_param']
        if blur_type == 0:
            raw_img = cv2.GaussianBlur(raw_img, (blur_param, blur_param), 0)
        elif blur_type == 1:
            raw_img = cv2.blur(raw_img, (blur_param, blur_param))
        elif blur_type == 2:
            raw_img = cv2.medianBlur(raw_img, blur_param)
        elif blur_type == 3:
            raw_img = cv2.boxFilter(raw_img, -1, (blur_param * 2, blur_param * 2))
        else:
            pass

        # Translate
        height, width = raw_img.shape[:2]
        M = np.float32([[1, 0, randomdict['translate_tw']],
                        [0, 1, randomdict['translate_th']]])
        raw_img = cv2.warpAffine(raw_img, M, (width, height), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask_img = cv2.warpAffine(mask_img, M, (width, height), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=0)
        # check_signal(mask_img, 'after translate')

        # Flip
        if randomdict['p_flip'] < randomdict['enhance']:
            raw_img = cv2.flip(raw_img, 1)
            mask_img = cv2.flip(mask_img, 1)
        # check_signal(mask_img, 'after flip')

        # Rotation
        raw_img = self.rotate(raw_img, randomdict['rotate_angle'], flags=cv2.INTER_CUBIC)
        mask_img = self.rotate(mask_img, randomdict['rotate_angle'], flags=cv2.INTER_NEAREST)
        # check_signal(mask_img, 'after rotation')

        # Pad
        pad_height = np.maximum(0, int(self._output_res[0] - raw_img.shape[0]))
        pad_width = np.maximum(0, int(self._output_res[1] - raw_img.shape[1]))
        if pad_height + pad_width > 0:
            raw_img = cv2.copyMakeBorder(raw_img, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT,
                                         value=0)
            mask_img = cv2.copyMakeBorder(mask_img, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT,
                                          value=0)
        # check_signal(mask_img, 'after padding')

        # Crop
        offset_h = np.random.randint(int(raw_img.shape[0] - self._output_res[0] + 1))
        offset_w = np.random.randint(int(raw_img.shape[1] - self._output_res[1] + 1))

        raw_img = raw_img[offset_h: offset_h + self._output_res[0], offset_w:offset_w + self._output_res[1], :]
        mask_img = mask_img[offset_h: offset_h + self._output_res[0], offset_w:offset_w + self._output_res[1]]

        mean = np.array([0.406, 0.456, 0.485])  # BGR
        std = np.array([0.225, 0.224, 0.229])
        raw_img = raw_img / 255.0
        raw_img -= mean
        raw_img /= std

        mask_img = np.expand_dims(mask_img, -1)
        _raw_img = im_to_torch(raw_img)
        _mask_img = im_to_torch(mask_img)
        # check_signal(_mask_img, 'after crop')
        return _raw_img, _mask_img

    #def data_normalization_for_val(self, raw_img, mask_img, edge_img):
    def data_normalization_for_val(self, raw_img, mask_img):
        # Pad
        pad_height = np.maximum(0, int(self._output_res[0] - raw_img.shape[0]))
        pad_width = np.maximum(0, int(self._output_res[1] - raw_img.shape[1]))
        if pad_height + pad_width > 0:
            raw_img = cv2.copyMakeBorder(raw_img, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT,
                                         value=0)
            mask_img = cv2.copyMakeBorder(mask_img, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT,
                                          value=0)


        # Crop
        offset_h = int((raw_img.shape[0] - self._output_res[0]) / 2)
        offset_w = int((raw_img.shape[1] - self._output_res[1]) / 2)

        raw_img = raw_img[offset_h: offset_h + self._output_res[0], offset_w:offset_w + self._output_res[1], :]
        mask_img = mask_img[offset_h: offset_h + self._output_res[0], offset_w:offset_w + self._output_res[1]]

        mean = np.array([0.406, 0.456, 0.485])  # BGR
        std = np.array([0.225, 0.224, 0.229])
        raw_img = raw_img / 255.0
        raw_img -= mean
        raw_img /= std

        mask_img = np.expand_dims(mask_img, -1)
        _raw_img = im_to_torch(raw_img)
        _mask_img = im_to_torch(mask_img)
        return _raw_img, _mask_img

    def __getitem__(self, item):
        name = self._name_list[item]

        img_path = os.path.normcase(os.path.join(self._data_path, 'image', name + self._img_format[0]))
       # seg_path = os.path.normcase(os.path.join(self._data_path, 'SegmentationClass', name + '_mask' + self._img_format[1]))

        seg_path = os.path.normcase(os.path.join(self._data_path, 'label', name + self._img_format[1]))


        # (H*W*3) 0-255
        raw_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # (H*W) 0,1
        mask_img = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
        # check_signal(mask_img, 'after cv2.imread', seg_path)

        if self._mode == 'train':
            # a sequence of augmentation operation for src_img
            output = self.data_transform(raw_img, mask_img)
            # raw:3*H*W mask:1*H*W
            _raw_img, _mask_img = output
            # check_signal(_mask_img, 'after data_transform', seg_path)

            entry = {
                'raw_image': _raw_img,
                'mask_image': _mask_img,
            }
        else:
            output = self.data_normalization_for_val(raw_img, mask_img)
            _raw_img, _mask_img = output

            entry = {
                'raw_image': _raw_img,
                'mask_image': _mask_img,
            }
        return entry

    def __len__(self):
        return len(self._name_list)


if __name__ == '__main__':
    train_dataset = Iris('train', data_path='/sdata/wenhui.ma/program/IrisUnet/data/forseg/', input_res=(256, 256))  # 连服务器必须用服务器文件路径，本地路径不行
    val_dataset = Iris('val', data_path='/sdata/wenhui.ma/program/IrisUnet/data/forseg/', input_res=(256, 256))
    print(train_dataset.__len__())


    training_data_loader = DataLoader(train_dataset, 6, shuffle=True, num_workers=4)

    val_data_loader = DataLoader(val_dataset, 6, shuffle=True, num_workers=4)

    # cpu or gpu?
    device_gpu = "cuda:0"
    if torch.cuda.is_available() and device_gpu is not None:
        device = torch.device(device_gpu)
    else:
        if not torch.cuda.is_available():
            print("hey man, buy a GPU!")
        device = torch.device("cpu")

    for ii, batch in enumerate(training_data_loader):
        raw_image = batch['raw_image'].to(device)
        mask_image = batch['mask_image'].to(device)
        print('load train data!')

    for ii, batch in enumerate(val_data_loader):
        raw_image = batch['raw_image'].to(device)
        mask_image = batch['mask_image'].to(device)
        print('load val data!')
