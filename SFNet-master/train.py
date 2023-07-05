import argparse
import logging
import os
import random
import sys
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate import evaluate
# from unet import UNet
from unet import SFNet
from utils.data_loading2 import BasicDataset
from utils.command_line import parse_args

dir_img = '/sdata/wenhui.ma/program/dataset/mobi_seg/train/image/'
dir_mask = '/sdata/wenhui.ma/program/dataset/mobi_seg/train/label/'
# dir_img = Path('/sdata/wenhui.ma/program/dataset/Iris-Seg/train/image/')
# dir_mask = Path('/sdata/wenhui.ma/program/dataset/Iris-Seg/train/label/')
dataset_name = 'mobi_seg'
# dataset_name = 'Iris-Seg'
dir_checkpoint = './' + f'{dataset_name}' + '_checkpoints/model/'


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 1,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # 1. Create dataset
    dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='SFNet', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 90, 150], gamma=0.1)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if args.classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0
    min_epoch_loss = 1000
    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        gc.collect()  # 垃圾回收
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                sclera_img, Iris_img, pupil_img, sclera_mask, Iris_mask, pupil_mask, mask, name = batch

                assert sclera_img.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {sclera_img.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'
                assert Iris_img.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {Iris_img.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'
                assert pupil_img.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {pupil_img.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                sclera_img = sclera_img.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                Iris_img = Iris_img.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                pupil_img = pupil_img.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                sclera_mask = sclera_mask.to(device=device, dtype=torch.long)  # torch.long即torch.int64
                Iris_mask = Iris_mask.to(device=device, dtype=torch.long)
                pupil_mask = pupil_mask.to(device=device, dtype=torch.long)
                mask = mask.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    mask_pred, mask_pred_sclera, mask_pred_Iris, mask_pred_pupil = model(sclera_img, Iris_img,
                                                                                         pupil_img)
                    if args.classes == 1:
                        loss = criterion(mask_pred.squeeze(1), mask.float())
                        loss += 0.5 * criterion(mask_pred_sclera.squeeze(1), sclera_mask.float())
                        # loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(mask_pred, mask)
                        loss += 0.3 * criterion(mask_pred_sclera, sclera_mask) + \
                                0.2 * criterion(mask_pred_Iris, Iris_mask) + \
                                0.1 * criterion(mask_pred_pupil, pupil_mask)
                        # loss += dice_loss(
                        #     F.softmax(masks_pred, dim=1).float(),
                        #     F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                        #     multiclass=True
                        # )

                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(sclera_img.shape[0])
                global_step += 1
                epoch_loss += loss.item()

                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_iou, val_loss = evaluate(model, val_loader, device, criterion,
                                                     dataset.mask_values, args)
                        scheduler.step(val_iou)

                        logging.info('Validation Iou score: {}'.format(val_iou))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Iou': val_iou,
                                'validation loss': val_loss,
                                'masks': {
                                    'true': wandb.Image(mask[0].float().cpu()),
                                    'pred': wandb.Image(mask_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'images': wandb.Image(sclera_img[0].cpu()),
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass

        if epoch_loss < min_epoch_loss:
            min_epoch_loss = epoch_loss
            if save_checkpoint:
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                state_dict = model.state_dict()
                state_dict['mask_values'] = dataset.mask_values
                torch.save(state_dict, dir_checkpoint + 'checkpoint_epoch{}.pth'.format(epoch))
                logging.info(f'Checkpoint {epoch} saved!')


if __name__ == '__main__':
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    args.classes = 4
    args.epochs = 200
    args.batch_size = 12
    model = SFNet(n_channels=3, n_classes=args.classes, deep_mask=True)
    model = model.to(memory_format=torch.channels_last)

    # logging.info(f'Network:\n'
    #              f'\t{model.n_channels} input channels\n'
    #              f'\t{model.n_classes} output channels (classes)\n'
    #              f'\tBilinear upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)

    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val,
            amp=args.amp
        )
