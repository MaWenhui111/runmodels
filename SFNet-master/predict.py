import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2
from torchvision import transforms
from torch.utils.data import DataLoader
from pathlib import Path

from utils.data_loading2 import BasicDataset
from unet import SFNet
from utils.command_line import parse_args
from evaluate import evaluate


# def predict_img(net,dataloader,device,scale_factor=1,out_threshold=0.5):
#     net.eval()
#     img, Iris, sclera, pupil = BasicDataset(None, full_img, is_mask=False)
#     Iris = torch.from_numpy(Iris)
#     sclera = torch.from_numpy(sclera)
#     pupil = torch.from_numpy(pupil)
#     Iris = Iris.unsqueeze(0)
#     sclera = sclera.unsqueeze(0)
#     pupil = pupil.unsqueeze(0)
#     Iris = Iris.to(device=device, dtype=torch.float32)
#     sclera = sclera.to(device=device, dtype=torch.float32)
#     pupil = pupil.to(device=device, dtype=torch.float32)
#
#     with torch.no_grad():
#         output = net(img)['output'].cpu()
#         # output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
#         if net.n_classes > 1:
#             mask = output.argmax(dim=1)
#         else:
#             mask = torch.sigmoid(output) > out_threshold
#
#     return mask[0].long().squeeze().numpy()
#
#
# def get_output_filenames(args):
#     def _generate_name(fn):
#         return f'{os.path.splitext(fn)[0]}_OUT.png'
#
#     return args.output or list(map(_generate_name, args.input))
#
#
#
#
#
# def get_args():
#     parser = argparse.ArgumentParser(description='Predict masks from input images')
#     parser.add_argument('--model', '-m',
#                         default='/sdata/wenhui.ma/program/models/SFNet-master/mobi_seg_checkpoints/best_model/checkpoint_epoch195.pth',
#                         metavar='FILE', help='Specify the file in which the model is stored')
#     parser.add_argument('--input', '-i', default='/sdata/wenhui.ma/program/dataset/Mobi-Seg/test/image/',
#                         metavar='INPUT', nargs='+', help='Filenames of input images')
#     parser.add_argument('--output', '-o',
#                         default='/sdata/wenhui.ma/program/models/SFNet-master/Mobi-Seg_predict/',
#                         metavar='OUTPUT', nargs='+', help='Filenames of output images')
#     parser.add_argument('--viz', '-v', action='store_true',
#                         help='Visualize the images as they are processed')
#     parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
#     parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
#                         help='Minimum probability value to consider a mask pixel white')
#     parser.add_argument('--scale', '-s', type=float, default=0.6,
#                         help='Scale factor for the input images')
#     parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
#     parser.add_argument('--classes', '-c', type=int, default=4, help='Number of classes')
#
#     return parser.parse_args()


if __name__ == '__main__':
    # args = get_args()
    args = parse_args()
    args.model = '/sdata/wenhui.ma/program/models/SFNet-master/mobi_seg_checkpoints/model/checkpoint_epoch196.pth'
    args.classes = 4
    args.save = True
    args.batch_size = 1
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    dir_img = '/sdata/wenhui.ma/program/dataset/mobi_seg/test/image/'
    dir_mask = '/sdata/wenhui.ma/program/dataset/mobi_seg/test/label/'
    dataset = BasicDataset(dir_img, dir_mask, args.scale)
    loader_args = dict(batch_size=args.batch_size, num_workers=os.cpu_count(), pin_memory=True)
    data_loader = DataLoader(dataset, shuffle=False, **loader_args)

    net = SFNet(n_channels=3, n_classes=args.classes)
    criterion = nn.CrossEntropyLoss() if net.n_classes > 1 else nn.BCEWithLogitsLoss()

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values')
    net.load_state_dict(state_dict, False)

    logging.info('Model loaded!')

    test_iou, test_loss = evaluate(net, data_loader, device=device,
                                   criterion=criterion, mask_values=mask_values, args=args)
    logging.info(f'Mask saved to {args.output}')
    #
    # in_files = os.listdir(args.input)
    # out_files = get_output_filenames(args)
    # if not os.path.exists(out_files):
    #     os.makedirs(out_files)
    #
    #
    # args.classes = 4
    # net = SFNet(n_channels=3, n_classes=args.classes)
    #
    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # logging.info(f'Loading model {args.model}')
    # logging.info(f'Using device {device}')
    #
    # net.to(device=device)
    # state_dict = torch.load(args.model, map_location=device)
    # mask_values = state_dict.pop('mask_values')
    # net.load_state_dict(state_dict, False)
    #
    # logging.info('Model loaded!')
    # ious = []
    # for i, filename in enumerate(in_files):
    #     logging.info(f'Predicting image {filename} ...')
    #     img = cv2.imread(args.input + filename)
    #     true_mask = cv2.imread(
    #         '/sdata/wenhui.ma/program/dataset/Mobi-Seg/test/label/' + os.path.splitext(filename)[0] + '.png')
    #     true_mask = BasicDataset.preprocess(mask_values, true_mask,  is_mask=True)
    #     mask = predict_img(net=net,
    #                        full_img=img,
    #                        scale_factor=args.scale,
    #                        out_threshold=args.mask_threshold,
    #                        device=device)

        # if not args.no_save:
        #     out_filename = out_files
        #     result = mask_to_image(mask, mask_values)
        #     result.save(out_filename + os.path.splitext(filename)[0] + '.png')
        #     true_pred = np.hstack((true_mask, np.asarray(result)))
        #     cv2.imwrite(args.output + 'true_pred/' + os.path.splitext(filename)[0] + '.png', true_pred)
        #     logging.info(f'Mask saved to {out_filename}')
        #
        # if args.viz:
        #     logging.info(f'Visualizing results for image {filename}, close to continue...')
        #     plot_img_and_mask(img, mask)

        # true_mask = np.asarray(true_mask)[:, :, 0]
        # if (true_mask == 255).any():
        #     true_mask = true_mask / 255
        # iou = per_class_mIoU(mask, true_mask)
        # ious.append(iou)
    with open(args.output + 'test_iou.txt', 'a+') as f:
        f.write(Path(args.model).name+'：'+str(test_iou) + "\n")
    print(Path(args.model).name+'：')
    print('\ttest iou:', test_iou)
    print('\ttest loss:', test_loss)
