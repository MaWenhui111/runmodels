import cv2
import numpy as np
import scipy.ndimage
import torch
from src.util.tools import to_torch
import torch.nn.functional as F
from PIL import Image


def calculate_landmarks(x):  # input x: N*1*h*w  return N*8*2
    """Estimate landmark location from heatmaps."""
    num_points, h, w = x.shape[-3], x.shape[-2], x.shape[-1]
    ref_xs, ref_ys = np.meshgrid(np.linspace(0, 1.0, num=w, endpoint=True),
                                 np.linspace(0, 1.0, num=h, endpoint=True),
                                 indexing='xy')

    ref_xs = np.reshape(ref_xs, [-1, h * w])  # 1*(h*w)
    ref_ys = np.reshape(ref_ys, [-1, h * w])

    ref_xs = to_torch(ref_xs).float()
    ref_ys = to_torch(ref_ys).float()
    ref_xs = ref_xs.to(x.device)  # to x's device
    ref_ys = ref_ys.to(x.device)

    # Assuming x: N x 8 x h x w (NCHW)
    beta = 1e2  # https://bouthilx.wordpress.com/2013/04/21/a-soft-argmax/
    x = x.reshape([-1, num_points, h * w])
    x = F.softmax(beta * x, dim=-1)
    lmrk_xs = (ref_xs * x).sum(dim=2)  # N*1
    lmrk_ys = (ref_ys * x).sum(dim=2)

    # Return to actual coordinates ranges
    return (
        int(lmrk_xs * (w - 1.0) + 0.5),
        int(lmrk_ys * (h - 1.0) + 0.5),
    )  # 1 x 2


def feature_split(img, feature_num=2):
    tmp = np.zeros((feature_num, *img.shape))  # this place was edited, src: *img.shape
    for idx in range(feature_num):
        tmp[idx, :, :] = 1 * (img == idx + 1)
    return tmp


def heatmap2img(heatmap):
    ret = np.zeros((heatmap.shape[1], heatmap.shape[2], 3))
    ret[:, :, 0] = heatmap[0, :, :] * 255
    ret[:, :, 1] = heatmap[1, :, :] * 255
    return ret


def img2heatmap(img, feature_num=2):
    ret = np.zeros((feature_num, img.shape[0], img.shape[1]))
    for i in range(feature_num):
        ret[i, :, :] = img[:, :, i] / 255
    return ret


def fuse_img(heatmap, raw_img, fuse=True):
    if fuse:
        # heatmap[2, :, :] = raw_img
        heatmap = heatmap + raw_img / 2
    return heatmap


def gauss_heatmap(img, threshold=1):
    ret = np.zeros_like(img)
    for idx in range(img.shape[0]):
        dist = cv2.distanceTransform((1 - img[idx]).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        ret[idx, :, :] = np.exp(-np.square(dist) / (2 * np.square(threshold))) * (1 * (dist < 3 * threshold))
    return ret.astype(np.float32)


def heatmapimg2pil(heatmap, raw_img):
    ret = heatmap2img(heatmap).astype(np.uint8)
    if raw_img is not None:
        ret[:, :, 2] = raw_img
    return Image.fromarray(ret)


def pil2heatmapimg(img):
    img = np.array(img)
    heatmap = img2heatmap(img)
    ret_img = img[:, :, 2]
    return heatmap, ret_img[np.newaxis, :, :]


def heatmap2log(heatmap, used=2):
    ret_map = torch.zeros(3, heatmap.shape[-2], heatmap.shape[-1])
    if used == 1:
        ret_map[0] = heatmap.detach()
    else:
        ret_map[:2] = heatmap.detach()
    return ret_map


def img_inverse_transform(raw_img, shape):
    r = raw_img.shape[1]
    if shape[0] > shape[1]:
        tmp_shape = (int(r * shape[0] / shape[1]), r)
        tmp_img = np.zeros((*tmp_shape, 3))
        offset = int((tmp_shape[0] - r) / 2)
        tmp_img[offset: r + offset, :, 0:2] = raw_img[:, :, 0:2]
    else:
        tmp_shape = (r, int(r * shape[1] / shape[0]))
        tmp_img = np.zeros((*tmp_shape, 3))
        offset = int((tmp_shape[1] - r) / 2)
        tmp_img[:, offset: r + offset, 0:2] = raw_img[:, :, 0:2]
    return scipy.ndimage.zoom(tmp_img, (shape[0] / tmp_shape[0], shape[1] / tmp_shape[1], 1))


def img_transform(raw_img, gt_img=None, mode='test', size=256, enhance=0.5):
    gt_img = heatmap2img(gt_img) if gt_img is not None else None

    slen = np.minimum(raw_img.shape[0], raw_img.shape[1])
    if (mode != 'test' and enhance > 0) or enhance == 1:
        mean = cv2.mean(raw_img)[0:3]

        # Resize
        if np.random.rand() < enhance:
            f = np.random.choice(np.maximum(1, size / slen) * np.arange(0.5, 2, 0.25))
        else:
            f = size / slen
        raw_img = cv2.resize(raw_img, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR)
        gt_img = cv2.resize(gt_img, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR) if gt_img is not None else None
        height, width = raw_img.shape[:2]

        # Smoothness
        if np.random.rand() < enhance:
            smooth_type = np.random.randint(4)
            if smooth_type != 0:
                smooth_param = 1 + 2 * np.random.randint(3)
                if smooth_type == 1:
                    raw_img = cv2.GaussianBlur(raw_img, (smooth_param, smooth_param), 0)
                elif smooth_type == 2:
                    raw_img = cv2.blur(raw_img, (smooth_param, smooth_param))
                elif smooth_type == 3:
                    raw_img = cv2.medianBlur(raw_img, smooth_param)
                elif smooth_type == 4:
                    raw_img = cv2.boxFilter(raw_img, -1, (smooth_param * 2, smooth_param * 2))

        # Translate
        if np.random.rand() < enhance:
            if enhance == 1:
                max_translate = width * 0.8
            else:
                max_translate = width * 0.3
            M = np.float32([[1, 0, round(np.random.uniform(-max_translate, max_translate))],
                            [0, 1, round(np.random.uniform(-max_translate, max_translate))]])
            raw_img = cv2.warpAffine(raw_img, M, (width, height), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=mean)
            gt_img = cv2.warpAffine(gt_img, M, (width, height), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=[0, 0, 0]) if gt_img is not None else None

        # Flip
        if np.random.rand() < enhance:
            raw_img = cv2.flip(raw_img, 1)
            gt_img = cv2.flip(gt_img, 1) if gt_img is not None else None

        # Rotation
        if np.random.rand() < enhance:
            angle = round(np.random.uniform(-60, 60))
            M = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), angle, 1)
            raw_img = cv2.warpAffine(raw_img, M, (width, height), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=mean)
            gt_img = cv2.warpAffine(gt_img, M, (width, height), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=[0, 0, 0]) if gt_img is not None else None

        # Check if we need to pad img to fit for crop_size
        pad_height = np.maximum(0, int((size - raw_img.shape[0]) / 2 + 1))
        pad_width = np.maximum(0, int((size - raw_img.shape[1]) / 2 + 1))
        if pad_height + pad_width > 0:
            raw_img = cv2.copyMakeBorder(raw_img, pad_height, pad_height, pad_width, pad_width, cv2.BORDER_CONSTANT,
                                         value=mean)
            gt_img = cv2.copyMakeBorder(gt_img, pad_height, pad_height, pad_width, pad_width, cv2.BORDER_CONSTANT,
                                        value=[0, 0, 0]) if gt_img is not None else None
        assert np.min(raw_img.shape[:2]) >= size
        if gt_img is not None:
            assert raw_img.shape[0] == gt_img.shape[0]
            assert raw_img.shape[1] == gt_img.shape[1]

        # Crop
        offset_h = np.random.randint(int((raw_img.shape[0] - size))) if raw_img.shape[0] != size else 0
        offset_w = np.random.randint(int((raw_img.shape[1] - size))) if raw_img.shape[1] != size else 0
        raw_img = raw_img[offset_h: offset_h + size, offset_w:offset_w + size, :]
        gt_img = gt_img[offset_h: offset_h + size, offset_w:offset_w + size] if gt_img is not None else None

    else:
        # Resize
        f = size / slen
        raw_img = cv2.resize(raw_img, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR)
        gt_img = cv2.resize(gt_img, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR) if gt_img is not None else None
        # CenterCrop
        offset_h, offset_w = int((raw_img.shape[0] - size) / 2), int((raw_img.shape[1] - size) / 2)
        raw_img = raw_img[offset_h: offset_h + size, offset_w:offset_w + size, :]
        gt_img = gt_img[offset_h: offset_h + size, offset_w:offset_w + size] if gt_img is not None else None

    raw_img = img2heatmap(raw_img, 3)
    gt_img = img2heatmap(gt_img) if gt_img is not None else None

    return raw_img, gt_img


def mask_transform(raw_img, iris_img=None, pupil_img=None, mode='train', enhance=0.5):
    raw_img_mean = cv2.mean(raw_img)[0:3]
    size = np.ceil(np.array(raw_img.shape[0:2]) / 32) * 32

    if mode == 'train':
        slen = np.minimum(raw_img.shape[0], raw_img.shape[1])

        # Resize
        if np.random.rand() < enhance:
            r = np.random.choice(np.arange(0.8, 1.2, 0.02))
            fh, fw = r, r
        else:
            fh = size[0] / slen
            fw = size[1] / slen
        raw_img = cv2.resize(raw_img, None, fx=fw, fy=fh, interpolation=cv2.INTER_LINEAR)
        iris_img = cv2.resize(iris_img, None, fx=fw, fy=fh,
                              interpolation=cv2.INTER_LINEAR) if iris_img is not None else None
        pupil_img = cv2.resize(pupil_img, None, fx=fw, fy=fh,
                               interpolation=cv2.INTER_LINEAR) if pupil_img is not None else None
        height, width = raw_img.shape[:2]

        # Smoothness
        if np.random.rand() < enhance:
            smooth_type = np.random.randint(4)
            if smooth_type != 0:
                smooth_param = 1 + 2 * np.random.randint(7)
                if smooth_type == 1:
                    raw_img = cv2.GaussianBlur(raw_img, (smooth_param, smooth_param), 0)
                elif smooth_type == 2:
                    raw_img = cv2.blur(raw_img, (smooth_param, smooth_param))
                elif smooth_type == 3:
                    raw_img = cv2.medianBlur(raw_img, smooth_param)
                elif smooth_type == 4:
                    raw_img = cv2.boxFilter(raw_img, -1, (smooth_param * 2, smooth_param * 2))

        # Translate
        if np.random.rand() < enhance:
            M = np.float32([[1, 0, round(np.random.uniform(-15, 15))],
                            [0, 1, round(np.random.uniform(-15, 15))]])
            raw_img = cv2.warpAffine(raw_img, M, (width, height), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=raw_img_mean)
            iris_img = cv2.warpAffine(iris_img, M, (width, height), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=0) if iris_img is not None else None
            pupil_img = cv2.warpAffine(pupil_img, M, (width, height), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=0) if iris_img is not None else None

        # Flip
        if np.random.rand() < enhance:
            raw_img = cv2.flip(raw_img, 1)
            iris_img = cv2.flip(iris_img, 1) if iris_img is not None else None
            pupil_img = cv2.flip(pupil_img, 1) if pupil_img is not None else None

        # Rotation
        if np.random.rand() < enhance:
            angle = round(np.random.uniform(-10, 10))
            M = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), angle, 1)
            raw_img = cv2.warpAffine(raw_img, M, (width, height), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=raw_img_mean)
            iris_img = cv2.warpAffine(iris_img, M, (width, height), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=0) if iris_img is not None else None
            pupil_img = cv2.warpAffine(pupil_img, M, (width, height), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=0) if pupil_img is not None else None

        # Check if we need to pad img to fit for crop_size
        pad_height = np.maximum(0, int((size[0] - raw_img.shape[0]) / 2 + 1))
        pad_width = np.maximum(0, int((size[1] - raw_img.shape[1]) / 2 + 1))
        if pad_height + pad_width > 0:
            raw_img = cv2.copyMakeBorder(raw_img, pad_height, pad_height, pad_width, pad_width, cv2.BORDER_CONSTANT,
                                         value=raw_img_mean)
            iris_img = cv2.copyMakeBorder(iris_img, pad_height, pad_height, pad_width, pad_width, cv2.BORDER_CONSTANT,
                                          value=0) if iris_img is not None else None
            pupil_img = cv2.copyMakeBorder(pupil_img, pad_height, pad_height, pad_width, pad_width, cv2.BORDER_CONSTANT,
                                           value=0) if pupil_img is not None else None
        assert (raw_img.shape[0]) >= size[0]
        assert (raw_img.shape[1]) >= size[1]
        if iris_img is not None:
            assert raw_img.shape[0] == iris_img.shape[0]
            assert raw_img.shape[1] == iris_img.shape[1]
            assert raw_img.shape[0] == pupil_img.shape[0]
            assert raw_img.shape[1] == pupil_img.shape[1]

        # Crop
        offset_h = np.random.randint(int((raw_img.shape[0] - size[0]))) if raw_img.shape[0] != size[0] else 0
        offset_w = np.random.randint(int((raw_img.shape[1] - size[1]))) if raw_img.shape[1] != size[1] else 0
        raw_img = raw_img[offset_h: int(offset_h + size[0]), offset_w:int(offset_w + size[1]), :]
        iris_img = iris_img[offset_h: int(offset_h + size[0]),
                   offset_w:int(offset_w + size[1])] if iris_img is not None else None
        pupil_img = pupil_img[offset_h: int(offset_h + size[0]),
                    offset_w:int(offset_w + size[1])] if pupil_img is not None else None
    else:

        pad_height = np.maximum(0, int(size[0] - raw_img.shape[0]))
        pad_width = np.maximum(0, int(size[1] - raw_img.shape[1]))
        if pad_height + pad_width > 0:
            raw_img = cv2.copyMakeBorder(raw_img, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT,
                                         value=raw_img_mean)
            iris_img = cv2.copyMakeBorder(iris_img, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT,
                                          value=0) if iris_img is not None else None
            pupil_img = cv2.copyMakeBorder(pupil_img, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT,
                                           value=0) if pupil_img is not None else None

    return img2heatmap(raw_img, 3), iris_img, pupil_img


def mask_inverse_transform(raw_img, shape):
    if len(raw_img.shape) == 3:
        raw_img = raw_img[0:shape[0], 0:shape[1], :]
    else:
        raw_img = raw_img[0:shape[0], 0:shape[1]]
    return raw_img


def draw_circle_ellipse(img, param, type='circle'):
    """

    :param img: Tensor (C*H*W)
    :param param: Tensor (6,) or (10,)
    :param type: string  'circle' or 'ellipse'
    :return: Tensor (C*H*W)
    """
    device = img.device
    img = img.cpu().numpy()
    param = param.cpu().numpy()
    img_shape = img.shape
    img_dtype = img.dtype
    try:
        if type == 'circle':
            img[2] = cv2.circle(img[2], center=(param[0], param[1]), radius=param[2],
                                color=(1, 0, 1), thickness=2)
            img[1] = cv2.circle(img[1], center=(param[3], param[4]), radius=param[5],
                                color=(1, 1, 0), thickness=2)
        elif type == 'ellipse':
            img[2] = cv2.ellipse(img[2], (param[0], param[1]), (param[2], param[3]), -np.rad2deg(param[4]), 0, 360,
                                 color=(1, 0, 1), thickness=2)
            img[1] = cv2.ellipse(img[1], (param[5], param[6]), (param[7], param[8]), -np.rad2deg(param[9]), 0, 360,
                                 color=(1, 1, 0), thickness=2)
    except BaseException as e:
        print(e)
        print(param)
        img = np.zeros(img_shape, dtype=img_dtype)
    finally:
        return torch.from_numpy(img).to(device)


def data_transform(raw_img, *other, mode='train', type='circle', enhance=0.5, size=416, randomdict=None):
    if len(other) > 0 and len(other[-1].shape) == 1:
        target_param = other[-1]
        mask_imgs = list(other[:-1])
    else:
        target_param = np.array((-100,) * 15, dtype=np.float32)
        mask_imgs = list(other)

    slen = np.maximum(raw_img.shape[0], raw_img.shape[1])
    raw_img_mean = (104.008, 116.669, 122.675)
    raw_img -= np.array(raw_img_mean, dtype=np.float32)

    if mode == 'train':
        randomdict = {
            'enhance': enhance,
            'p_resize': np.random.rand(),
            'resize': np.random.choice(np.arange(0.9, 1.1, 0.02)),
            'p_blur': np.random.rand(),
            'blur_type': np.random.randint(4),
            'blur_param': 1 + 2 * np.random.randint(7),
            'p_translate': np.random.rand(),
            'translate_tw': round(np.random.uniform(-15, 15)),
            'translate_th': round(np.random.uniform(-15, 15)),
            'p_flip': np.random.rand(),
            'p_crop': np.random.rand(),
            'offset_h': 0,
            'offset_w': 0
        } if randomdict is None else randomdict

        # Resize
        if randomdict['p_resize'] < randomdict['enhance']:
            f = (size / slen) * randomdict['resize']
        else:
            f = size / slen
        raw_img = cv2.resize(raw_img, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR)
        for i in range(len(mask_imgs)):
            mask_imgs[i] = cv2.resize(mask_imgs[i], None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
        if type == 'circle':
            target_param *= f
        else:
            target_param[0:4] *= f
            target_param[5:9] *= f
        height, width = raw_img.shape[:2]

        # Blur
        if randomdict['p_blur'] < randomdict['enhance']:
            blur_type = randomdict['blur_type']
            if blur_type != 0:
                blur_param = randomdict['blur_param']
                if blur_type == 1:
                    raw_img = cv2.GaussianBlur(raw_img, (blur_param, blur_param), 0)
                elif blur_type == 2:
                    raw_img = cv2.blur(raw_img, (blur_param, blur_param))
                # elif blur_type == 3:
                #     raw_img = cv2.medianBlur(raw_img, blur_param)
                elif blur_type == 4:
                    raw_img = cv2.boxFilter(raw_img, -1, (blur_param * 2, blur_param * 2))

        # Translate
        if randomdict['p_translate'] < randomdict['enhance']:
            M = np.float32([[1, 0, randomdict['translate_tw']],
                            [0, 1, randomdict['translate_th']]])
            raw_img = cv2.warpAffine(raw_img, M, (width, height), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=raw_img_mean)
            for i in range(len(mask_imgs)):
                mask_imgs[i] = cv2.warpAffine(mask_imgs[i], M, (width, height), flags=cv2.INTER_LINEAR,
                                              borderMode=cv2.BORDER_CONSTANT,
                                              borderValue=255)
            if type == 'circle':
                target_param[0] += randomdict['translate_tw']
                target_param[1] += randomdict['translate_th']
                target_param[3] += randomdict['translate_tw']
                target_param[4] += randomdict['translate_th']
            else:
                target_param[0] += randomdict['translate_tw']
                target_param[1] += randomdict['translate_th']
                target_param[5] += randomdict['translate_tw']
                target_param[6] += randomdict['translate_th']

        # Flip
        if randomdict['p_resize'] < randomdict['enhance']:
            raw_img = cv2.flip(raw_img, 1)
            for i in range(len(mask_imgs)):
                mask_imgs[i] = cv2.flip(mask_imgs[i], 1)
            if type == 'circle':
                target_param[0] = width - target_param[0]
                target_param[3] = width - target_param[3]
            else:
                target_param[0] = width - target_param[0]
                target_param[5] = width - target_param[5]
                target_param[4] = np.pi - target_param[4]
                target_param[9] = np.pi - target_param[9]

        # Pad
        pad_height = np.maximum(0, int(size - raw_img.shape[0]))
        pad_width = np.maximum(0, int(size - raw_img.shape[1]))
        if pad_height + pad_width > 0:
            raw_img = cv2.copyMakeBorder(raw_img, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=raw_img_mean)
            for i in range(len(mask_imgs)):
                mask_imgs[i] = cv2.copyMakeBorder(mask_imgs[i], 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT,
                                                  value=255)

        # Crop
        offset_h = np.random.randint(int((raw_img.shape[0] - size))) if (
                raw_img.shape[0] != size and randomdict['p_crop'] < randomdict['enhance']) else 0
        offset_w = np.random.randint(int((raw_img.shape[1] - size))) if (
                raw_img.shape[1] != size and randomdict['p_crop'] < randomdict['enhance']) else 0
        randomdict['offset_h'], randomdict['offset_w'] = offset_h, offset_w
        raw_img = raw_img[offset_h: offset_h + size, offset_w:offset_w + size, :]
        for i in range(len(mask_imgs)):
            mask_imgs[i] = mask_imgs[i][offset_h: offset_h + size, offset_w:offset_w + size]
        if type == 'circle':
            target_param[0] -= offset_w
            target_param[1] -= offset_h
            target_param[3] -= offset_w
            target_param[4] -= offset_h
        else:
            target_param[0] -= offset_w
            target_param[1] -= offset_h
            target_param[5] -= offset_w
            target_param[6] -= offset_h
    else:
        f = size / slen
        raw_img = cv2.resize(raw_img, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR)
        for i in range(len(mask_imgs)):
            mask_imgs[i] = cv2.resize(mask_imgs[i], None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
        if type == 'circle':
            target_param *= f
        else:
            target_param[0:4] *= f
            target_param[5:9] *= f

        pad_height = np.maximum(0, int(size - raw_img.shape[0]))
        pad_width = np.maximum(0, int(size - raw_img.shape[1]))
        if pad_height + pad_width > 0:
            raw_img = cv2.copyMakeBorder(raw_img, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=raw_img_mean)
            for i in range(len(mask_imgs)):
                mask_imgs[i] = cv2.copyMakeBorder(mask_imgs[i], 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT,
                                                  value=255)
        raw_img = raw_img[:size, :size, :]
        for i in range(len(mask_imgs)):
            mask_imgs[i] = mask_imgs[i][:size, :size]

    if len(target_param) < 12:
        mask_imgs.append(target_param)
    mask_imgs.insert(0, raw_img)
    mask_imgs.append(randomdict)

    return tuple(mask_imgs)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    fig = plt.figure()
    raw_img = cv2.imread('path',
                         cv2.IMREAD_COLOR).astype(np.float32)
    iris_img = cv2.imread('path',
                          cv2.IMREAD_UNCHANGED)
    pupil_img = cv2.imread('path',
                           cv2.IMREAD_UNCHANGED)
    mask_img = cv2.imread('path',
                          cv2.IMREAD_UNCHANGED)
    target_param_c = np.array((167.50, 155, 48, 50.45, 173.00, 152.47, 10.77)).astype(np.float32)
    target_param_e = np.array((167.5, 155.5, 53.99, 46.91, 1.80, 173.00, 152.49, 12.34, 9.25, 1.91)).astype(
        np.float32)

    other = (iris_img.copy(), pupil_img.copy(), mask_img.copy(), target_param_e.copy())
    t_raw_img, other = data_transform(raw_img.copy(), other, type='ellipse', mode='train', size=256, enhance=1)
    t_iris_img, t_pupil_img, t_mask_img, t_target_param = other
    print(t_raw_img.shape, t_iris_img.shape, t_pupil_img.shape, t_mask_img.shape)

    ax1 = fig.add_subplot(241)
    # raw_img = cv2.circle(raw_img, center=(target_param_c[0], target_param_c[1]), radius=target_param_c[2],
    #                      color=(0, 255, 0), thickness=2)
    # raw_img = cv2.circle(raw_img, center=(target_param_c[3], target_param_c[4]), radius=target_param_c[5],
    #                      color=(0, 0, 255), thickness=2)
    raw_img = cv2.ellipse(raw_img, (target_param_e[0], target_param_e[1]), (target_param_e[2], target_param_e[3]),
                          -np.rad2deg(target_param_e[4]), 0, 360, color=(0, 255, 0), thickness=2)
    raw_img = cv2.ellipse(raw_img, (target_param_e[5], target_param_e[6]), (target_param_e[7], target_param_e[8]),
                          -np.rad2deg(target_param_e[9]), 0, 360, color=(0, 0, 255), thickness=2)
    ax1.imshow(cv2.cvtColor(raw_img.astype(np.uint8), cv2.COLOR_BGR2RGB))
    ax2 = fig.add_subplot(242)
    ax2.imshow(iris_img.astype(np.uint8), cmap='gray')
    ax3 = fig.add_subplot(243)
    ax3.imshow(pupil_img.astype(np.uint8), cmap='gray')
    ax4 = fig.add_subplot(244)
    ax4.imshow(mask_img.astype(np.uint8))
    ax5 = fig.add_subplot(245)
    # t_raw_img = cv2.circle(t_raw_img, center=(t_target_param[0], t_target_param[1]), radius=t_target_param[2],
    #                        color=(0, 255, 0), thickness=2)
    # t_raw_img = cv2.circle(t_raw_img, center=(t_target_param[3], t_target_param[4]), radius=t_target_param[5],
    #                        color=(0, 0, 255), thickness=2)
    t_raw_img = cv2.ellipse(t_raw_img, (t_target_param[0], t_target_param[1]), (t_target_param[2], t_target_param[3]),
                            np.rad2deg(t_target_param[4]), 0, 360, color=(0, 255, 0), thickness=2)
    t_raw_img = cv2.ellipse(t_raw_img, (t_target_param[5], t_target_param[6]), (t_target_param[7], t_target_param[8]),
                            np.rad2deg(t_target_param[9]), 0, 360, color=(0, 0, 255), thickness=2)
    ax5.imshow(cv2.cvtColor(t_raw_img.astype(np.uint8), cv2.COLOR_BGR2RGB))
    ax6 = fig.add_subplot(246)
    ax6.imshow(t_iris_img.astype(np.uint8), cmap='gray')
    ax7 = fig.add_subplot(247)
    ax7.imshow(t_pupil_img.astype(np.uint8), cmap='gray')
    ax8 = fig.add_subplot(248)
    ax8.imshow(t_mask_img.astype(np.uint8))

    plt.show()
