import logging
from functools import partial
from multiprocessing import Pool
import os
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import cv2


# def load_image(filename):
#     ext = splitext(filename)[1]  # 返回filename扩展名
#     if ext == '.npy':
#         return Image.fromarray(np.load(filename))
#     elif ext in ['.pt', '.pth']:
#         return Image.fromarray(torch.load(filename).numpy())
#     else:
#         return Image.open(filename)


def unique_mask_values(idx, mask_dir):
    mask_file = list(mask_dir.glob(idx + '.*'))[0]
    mask_file = str(mask_file)
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    # mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, deep_mask=True, num_classes=4):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.deep_mask = deep_mask
        self.num_classes = num_classes
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if
                    isfile(join(images_dir, file)) and not file.startswith('.')]  # 返回图像名
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, img, scale, is_mask):
        h, w = img.shape[:2]
        if scale != 1:
            # newW, newH = int(scale * w), int(scale * h)
            # assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
            # pil_img = pil_img.resize((newW, newH), resample=Image.Resampling.NEAREST if is_mask else Image.Resampling.BICUBIC)
            w, h = int(scale * w), int(scale * h)
            assert w > 0 and h > 0, 'Scale is too small, resized images would have no pixel'
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NN if is_mask else cv2.INTER_CUBIC)

        if is_mask:
            # mask = np.zeros((newH, newW), dtype=np.int64)
            mask = np.zeros((h, w), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        # mask = load_image(mask_file[0])
        # img = load_image(img_file[0])
        mask = cv2.imread(str(mask_file[0]), cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(str(img_file[0]))
        assert img.shape[:2] == mask.shape[:2], \
            f'Image and mask {name} should be the same size, but are {img.shape} and {mask.shape}'

        # 灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.expand_dims(gray, axis=0)  # 将灰度图的维度扩为3维
        pupil = np.concatenate((gray, gray, gray), axis=0)

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)  # 由h,w,c变为c,h,w
        Iris = img.copy()
        sclera = img.copy()

        # R通道
        Iris[:, :, 0] = Iris[:, :, 2]
        Iris[:, :, 1] = Iris[:, :, 2]

        # G通道
        sclera[:, :, 0] = sclera[:, :, 1]
        sclera[:, :, 2] = sclera[:, :, 1]

        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)
        num_mask = np.zeros((mask.shape[0], mask.shape[1], self.num_classes), dtype=np.int64)
        for i in range(self.num_classes):
            num_mask[:, :, i] += (mask == i).astype(np.uint8)  # 将每种label分开成不同的通道

        sclera_mask = num_mask[:, :, 1]
        Iris_mask = num_mask[:, :, 2]
        pupil_mask = num_mask[:, :, 3]

        return sclera, Iris, pupil, sclera_mask, Iris_mask, pupil_mask, mask
        # if self.deep_mask:
        #     mask2 = self.preprocess(self.mask_values, mask, self.scale / 4, is_mask=True)
        #
        #     return {
        #         'image': torch.as_tensor(img.copy()).float().contiguous(),
        #         'mask1': torch.as_tensor(mask1.copy()).long().contiguous(),
        #         'mask2': torch.as_tensor(mask2.copy()).long().contiguous()
        #     }
        # else:
        #     return torch.as_tensor(img.copy()).float().contiguous(), torch.as_tensor(mask1.copy()).long().contiguous(),


if __name__ == '__main__':
    img_scale = 1
    batch_size = 24
    dir_img = '/sdata/wenhui.ma/program/dataset/mobi_seg/train/image/'
    dir_mask = '/sdata/wenhui.ma/program/dataset/mobi_seg/train/label/'

    dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    dataset_loader = DataLoader(dataset, shuffle=True, **loader_args)

    for batch in dataset_loader:
        sclera, Iris, pupil, sclera_mask, Iris_mask, pupil_mask, mask = batch
    print(sclera.shape, Iris.shape, pupil.shape, sclera_mask.shape, Iris_mask.shape, pupil_mask.shape, mask.shape)
