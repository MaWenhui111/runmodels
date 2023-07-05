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


def load_image(filename):
    ext = splitext(filename)[1]  # 返回filename扩展名
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = '', deep_mask=True):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.deep_mask = deep_mask

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if
                    isfile(join(images_dir, file)) and not file.startswith('.')]  # 返回图像名
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        if scale != 1:
            # newW, newH = int(scale * w), int(scale * h)
            # assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
            # pil_img = pil_img.resize((newW, newH), resample=Image.Resampling.NEAREST if is_mask else Image.Resampling.BICUBIC)
            w, h = int(scale * w), int(scale * h)
            assert w > 0 and h > 0, 'Scale is too small, resized images would have no pixel'
            pil_img = pil_img.resize((w, h),
                                     resample=Image.Resampling.NEAREST if is_mask else Image.Resampling.BICUBIC)
        img = np.asarray(pil_img)

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
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask1 = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)
        if self.deep_mask:
            mask2 = self.preprocess(self.mask_values, mask, self.scale / 4, is_mask=True)

            return {
                'image': torch.as_tensor(img.copy()).float().contiguous(),
                'mask1': torch.as_tensor(mask1.copy()).long().contiguous(),
                'mask2': torch.as_tensor(mask2.copy()).long().contiguous()
            }
        else:
            return torch.as_tensor(img.copy()).float().contiguous(), torch.as_tensor(mask1.copy()).long().contiguous(),


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='')


if __name__ == '__main__':
    img_scale = 1
    val_percent = 0.1
    batch_size = 24
    dir_img = Path('/sdata/wenhui.ma/program/models/EyeSeg-master2/EyeSeg-master/data/Iris-Seg/train/image')
    dir_mask = Path('/sdata/wenhui.ma/program/models/EyeSeg-master2/EyeSeg-master/data/Iris-Seg/train/label')
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    for batch in val_loader:
        images, true_masks1, true_masks2 = batch['image'], batch['mask1'], batch['mask2']
    print(images.shape, true_masks1.shape, true_masks2.shape)

    # mask = Image.open('/sdata/wenhui.ma/program/models/EyeSeg-master2/EyeSeg-master/data/Iris-Seg/train/label/999.png')
    # mask_values = [[0, 0, 0], [255, 255, 255]]
    # mask = BasicDataset.preprocess(mask_values, mask, 1, True)

