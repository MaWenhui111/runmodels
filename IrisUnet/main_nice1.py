import logging
import random
import time

import numpy as np
import torch.backends.cudnn
import torch.optim
from torch.utils.data import DataLoader

import src.datasources as datasources
import src.models as models
from config_nice1 import DefaultConfig
from src.util.osutils import isfile, join

try:
    # from tensorboardX import SummaryWriter
    from torch.utils.tensorboard import SummaryWriter

    is_tensorboard_available = True
except Exception:
    is_tensorboard_available = False

torch.backends.cudnn.benchmark = True

logging.basicConfig(
    format='[%(asctime)s %(name)s %(levelname)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)


def main(pretrained_model):
    if pretrained_model and isfile(pretrained_model):
        logger.info("=> loading checkpoint '{}'".format(pretrained_model))
        checkpoint = torch.load(pretrained_model)
        cfg = checkpoint['cfg']
    else:
        logger.warning("=> no checkpoint found at '{}', use the default configure instead.".format(pretrained_model))
        cfg = DefaultConfig()


    # 可以更新cfg的内容
    new_config = {'isTrain': True}
    cfg.parse(new_config)

    # 配置保存模型名字
    train_name = time.strftime("%m%d_%H%M%S", time.localtime())
    train_name = ('Iris_' + cfg.dataset + '_i%dx%d_' + train_name + '_' + cfg.method) % (
        cfg.input_res[0], cfg.input_res[1],
    )
    cfg.name = train_name
    cfg.global_step = 0   # 在config_nice1.py里没找到
    cfg.lr = cfg.initial_lr  # 当前学习率    呃呃呃呃，cfg.lr也没找到
    if cfg.log_interval is None:
        cfg.log_interval = int(np.ceil(cfg.max_epochs * 0.02))

    # Set Random Seed
    #seed = cfg.seed
    seed = 17
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


    # Set Tensorboard
    if cfg.visible and is_tensorboard_available:
        log_writer = SummaryWriter(join(cfg.checkpoint, cfg.name, "log"))  # create log dir
        #log_writer = SummaryWriter(log_dir = '/root/tf-logs/')
        log_writer.add_text('cur_cfg', cfg.__str__())
        with open(join(cfg.checkpoint, cfg.name, "log", "config.txt"), 'w') as f:
            f.write(cfg.__str__())
    else:
        log_writer = None

    # Set Device
    if torch.cuda.is_available() and cfg.device is not None:
        device = torch.device(cfg.device)
    else:
        if not torch.cuda.is_available():
            logger.warning("hey man, buy a GPU!!!!!!")
        device = torch.device("cpu")

    # Create Model
    logger.info("==> creating model '{}'".format(cfg.model_arch))
    model = models.__dict__[cfg.model_arch](cfg, logger)   #加载预训练模型名
    from iris_train_val import validate, train

    # Load Model
    if pretrained_model and isfile(pretrained_model):
        model.load_networks(checkpoint)
        logger.info("=> loaded model from checkpoint '{}'"
                    .format(pretrained_model))
    else:
        logger.info("=> initialize the weight from scratch.")

    model.print_networks(False)
    model.to(device)

    # DataLoader
    train_dataset = datasources.Iris('train', data_path=cfg.root_path, input_res=cfg.input_res,
                                     img_format=cfg.img_format)
    val_dataset = datasources.Iris('val', data_path=cfg.root_path, input_res=cfg.input_res, img_format=cfg.img_format)

    train_loader = DataLoader(train_dataset, batch_size=cfg.train_batch, collate_fn=datasources.my_collate_fn,
                              shuffle=True, num_workers=cfg.num_workers, pin_memory=False)

    val_loader = DataLoader(val_dataset, batch_size=cfg.val_batch, collate_fn=datasources.my_collate_fn,
                            shuffle=True, num_workers=cfg.num_workers, pin_memory=False)

    cfg.max_iter = int(len(train_dataset)/cfg.train_batch * cfg.max_epochs)
    logger.info(cfg)

    # Test0
    logger.info('Run test before start training.')
    val_loss = validate(0, model, device, val_loader, logger, log_writer, cfg)

    # Start!
    logger.info("Start training!")
    min_val_loss = val_loss

    for epoch in range(1, cfg.max_epochs + 1):

        # train for one epoch
        train(epoch, model, device, train_loader, logger, log_writer, cfg, snapshot=200)

        # evaluate on validation set
        val_loss = validate(epoch, model, device, val_loader, logger, log_writer, cfg)

        # update lr
        new_lr = model.update_learning_rate_with_policy(val_loss)
        if log_writer is not None:
            log_writer.add_scalar('Train/LearningRate', new_lr, epoch)

        if val_loss < min_val_loss:
            model.save_networks('best', cfg, device)

        if ((cfg.log_interval > 0) and (epoch % cfg.log_interval == 0)) or \
                (epoch == cfg.max_epochs):
            model.save_networks(epoch, cfg, device)

    # if log_writer is not None:
    #     # save scales
    #     out_path = join(cfg.checkpoint, cfg.name, 'log', 'all_scalars.json')
    #     log_writer.export_scalars_to_json(out_path)


if __name__ == '__main__':
    # load_model_path = './checkpoints/***.pth'
    load_model_path = None
    main(load_model_path)
