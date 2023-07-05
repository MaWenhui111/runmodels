import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import Eye_UNet
from utils.utils import plot_img_and_mask, per_class_mIoU


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)['output'].cpu()
        # output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v
        # if i == 1:
        #     out[mask == i] = 78
        # elif i == 2:
        #     out[mask == i] = 178
        # elif i == 3:
        #     out[mask == i] = 255
        # else:
        #     out[mask == i] = 0
    return Image.fromarray(out)


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m',
                        default='/sdata/wenhui.ma/program/models/Pytorch-UNet-master/Iris-Seg_checkpoints/best_model/checkpoint_epoch194.pth',
                        metavar='FILE', help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', default='/sdata/wenhui.ma/program/dataset/Iris-Seg/test/image/',
                        metavar='INPUT', nargs='+', help='Filenames of input images')
    parser.add_argument('--output', '-o',
                        default='/sdata/wenhui.ma/program/models/Pytorch-UNet-master/Iris-Seg_predict/',
                        metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1.0,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = os.listdir(args.input)
    out_files = get_output_filenames(args)

    args.classes = 2
    net = Eye_UNet(n_channels=3, n_classes=args.classes)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict, False)

    logging.info('Model loaded!')
    ious = []
    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(args.input + filename)
        true_mask = Image.open(
            '/sdata/wenhui.ma/program/dataset/Iris-Seg/test/label/' + os.path.splitext(filename)[0] + '.png')
        true_mask = np.asarray(true_mask)
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_filename = out_files
            result = mask_to_image(mask, mask_values)
            result.save(out_filename + os.path.splitext(filename)[0] + '.png')
            true_pred = np.hstack((true_mask, np.asarray(result)))
            cv2.imwrite(args.output+'true_pred/'+os.path.splitext(filename)[0] + '.png', true_pred)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)

        true_mask = np.asarray(true_mask)[:, :, 0]
        if (true_mask==255).any():
            true_mask = true_mask/255
        mask = torch.as_tensor(mask.copy())
        true_mask = torch.as_tensor(true_mask.copy())
        iou = per_class_mIoU(mask, true_mask)
        ious.append(iou)
    with open(args.output+'test_iou.txt', 'a+') as f:
        f.write(str(np.average(ious))+ "\n")
    print(np.average(ious))
