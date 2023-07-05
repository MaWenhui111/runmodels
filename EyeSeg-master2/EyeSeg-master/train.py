import sys
import os
import pickle
from torch.utils.data import DataLoader
from contextlib import redirect_stdout
import torch
import torch.nn as nn
from torchsummary import summary
import multiprocessing as mp
import numpy as np
import cv2
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt
from data.dataloader3 import TorchData
from lib2 import args, flatten, make_one_hot, get_predictions, \
    per_class_mIoU, torch_validate, labelid_to_color
from model import EyeSeg
from losses import GeneralizedDiceLoss, EntropyLoss, \
    DiceLoss, SurfaceLoss, CrossEntropyLoss2d


def torch_main():
    # 使用多个GPU来加速训练
    device = torch.device('cuda:0')
    # model = torch.nn.DataParallel(EyeSeg( args.INPUT_SHAPE[-1],
    #                                         args.NUM_CLASSES,
    #                                         args.FILTERS,
    #                                         args.DROPOUT_RATE,
    #                                     ),
    #                                 device_ids=list(range(torch.cuda.device_count())),
    #                             )
    model = EyeSeg(args.INPUT_SHAPE[-1], args.NUM_CLASSES, args.FILTERS, args.DROPOUT_RATE)
    model.to(device)
    summary(model, (args.INPUT_SHAPE[-1], args.INPUT_SHAPE[0], args.INPUT_SHAPE[1]))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.LEARN_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    categorical_loss = CrossEntropyLoss2d()  # 交叉熵损失
    surface_loss = SurfaceLoss()  # 表面损失
    entropy = EntropyLoss()  # 熵损失
    dice_loss = GeneralizedDiceLoss()
    training_data = TorchData(args)
    validation_data = TorchData(args, mode='val')
    # unlabeled_data = TorchData(args, mode='semi')
    total_train_data = len(training_data) // args.BATCH_SIZE
    train = DataLoader(training_data, batch_size=args.BATCH_SIZE, shuffle=True, num_workers=args.WORKERS,
                       pin_memory=True)
    val = DataLoader(validation_data, batch_size=args.BATCH_SIZE, shuffle=False, num_workers=args.WORKERS,
                     pin_memory=True)
    # ul_train = DataLoader(unlabeled_data, batch_size=args.BATCH_SIZE, shuffle=True, num_workers=args.WORKERS, pin_memory=True)
    current_lr = args.LEARN_RATE
    os.makedirs(args.SAVE_FOLDER + args.MODEL_NAME, exist_ok=True)
    train_file = args.SAVE_FOLDER + args.MODEL_NAME + '/training_status.txt'
    with open(args.SAVE_FOLDER + args.MODEL_NAME + '/model_summary.txt', 'w') as f:
        with redirect_stdout(f):  # 将summary的输出写到文件f里
            summary(model, (args.INPUT_SHAPE[-1], args.INPUT_SHAPE[0], args.INPUT_SHAPE[1]))
    start = args.EPOCH_START
    model.train()

    for epoch in range(start, args.EPOCHS):
        ious = list()
        train_loss = list()
        gc.collect()  # 垃圾回收
        # ul_train_data = iter(ul_train)
        for batch_num, batch in enumerate(train):
            optimizer.zero_grad()  # 清空之前的梯度
            input_image, ground_truth, one_hot, spatial_gt, distMap, name = batch
            # loss_ul = None
            # if args.MODE == 'semi':
            #     try:
            #         input_ul_img, _, _, _, _, name_ul = next(ul_train_data)
            #     except:
            #         input_ul_img, name_ul = None, None
            #     if input_ul_img is not None:
            #         ul_in = input_ul_img.to(device)
            #         output_ul = model(ul_in)
            #         loss_ul = torch.mean(entropy(output_ul))
            data_in = input_image.to(device)
            output = model(data_in)
            cce = categorical_loss(output.to(device), ground_truth.to(device).long()) * (
                        torch.from_numpy(np.ones(spatial_gt.shape)).to(torch.float32).to(device) + (spatial_gt).to(
                    torch.float32).to(device))
            loss = torch.mean(dice_loss(output.to(device), ground_truth.to(device).long()))
            loss = torch.mean(cce) + loss
            predict = get_predictions(output)  # 在通道维上的最大值索引
            iou = per_class_mIoU(predict, ground_truth)

            ious.append(iou)

            # if args.MODE == 'semi' and loss_ul is not None:
            #     loss += loss_ul

            train_loss.append(loss.detach().item())
            # if (batch_num+1)%10 == 0 and loss_ul is not None:
            #     active_log = 'Epoch:{} [{}/{}], Loss: {:.3f}, UL_Loss: {:.3f}'.format(epoch+1,batch_num+1,total_train_data,loss.detach().item(), loss_ul.detach().item())
            #     print(active_log)
            #     with open(train_file, 'a+') as training_file:
            #         training_file.write(active_log + '\n')
            if (batch_num + 1) % 10 == 0:
                active_log = 'Epoch:{} [{}/{}], Loss: {:.3f}'.format(epoch + 1, batch_num + 1, total_train_data,
                                                                     loss.detach().item())
                print(active_log)
                with open(train_file, 'a+') as training_file:
                    training_file.write(active_log + '\n')
            loss.backward()
            optimizer.step()  # 更新网络参数
        mIoU, validation_loss = torch_validate(val, model, device, categorical_loss, dice_loss, surface_loss)
        epoch_end = 'Epoch:{}, Train mIoU: {:.4f}, Train Loss: {:.3f}'.format(epoch + 1, np.average(ious),
                                                                              np.average(train_loss))
        validation_results = 'Validation mIoU: {:.4f}, Validation Loss: {:.3f}'.format(mIoU, validation_loss)
        print(epoch_end)
        print(validation_results)
        with open(train_file, 'a+') as training_file:
            training_file.write(epoch_end + ' ' + validation_results + '\n')
        if epoch + 1 >= args.LOG_EPOCH:
            os.makedirs(args.SAVE_FOLDER + args.MODEL_NAME + '/epoch_' + str(epoch + 1) + '/', exist_ok=True)
            os.makedirs(args.SAVE_FOLDER + args.MODEL_NAME + '/epoch_' + str(epoch + 1) + '/' + 'hstack_imgs/',
                        exist_ok=True)
            os.makedirs(args.SAVE_FOLDER + args.MODEL_NAME + '/epoch_' + str(epoch + 1) + '/' + 'model/', exist_ok=True)
            os.makedirs(args.SAVE_FOLDER + args.MODEL_NAME + '/epoch_' + str(epoch + 1) + '/' + 'test_results/',
                        exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }
                , args.SAVE_FOLDER + args.MODEL_NAME + '/epoch_' + str(epoch + 1) + '/' + 'model/model_save.pkl')
            with torch.no_grad():
                for batch_n, data in enumerate(val):
                    if (batch_n + 1) % 10 == 0:
                        input_image, ground_truth, one_hot, spatial_gt, distMap, name = data
                        data_in = input_image.to(device)
                        output = model(data_in)
                        predict = get_predictions(output)
                        for idx in range(len(input_image)):
                            orig_im = np.squeeze(np.array(input_image[idx].cpu().numpy(), dtype=np.uint8))
                            orig_im = cv2.cvtColor(orig_im, cv2.COLOR_GRAY2BGR)
                            gt = labelid_to_color(np.array(ground_truth[idx].cpu().numpy(), dtype=np.uint8))
                            pr = labelid_to_color(predict[idx].cpu().numpy())
                            stack = np.hstack([orig_im, gt, pr])
                            plt.imsave(args.SAVE_FOLDER + args.MODEL_NAME + '/epoch_' + str(
                                epoch + 1) + '/' + 'hstack_imgs/' + str(name[idx]) + '.png', stack)
        scheduler.step(validation_loss)
        if current_lr > optimizer.param_groups[0]['lr']:
            with open(train_file, 'a+') as training_file:
                training_file.write(
                    'Learning Rate Decay on Epoch: {}\nLearning Rate Previous: {}\nLearning Rate Current: {}\n'.format(
                        epoch + 1, current_lr, optimizer.param_groups[0]['lr']))
                current_lr = optimizer.param_groups[0]['lr']


if __name__ == '__main__':
    args.FRAMEWORK = 'torch'
    if args.FRAMEWORK.lower().strip() == 'torch':
        torch_main()
