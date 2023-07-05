import time
import torch.optim
import torch.utils.data

from src.util.tools import AverageMeter


def train(epoch, model, device, train_loader, logger, log_writer, cfg, snapshot=200):
    logger.info('Train {}'.format(epoch))

    # switch to train mode
    model.train()
    # for refineNet only
   # model.freeze_bn()

    epoch_seg_loss = AverageMeter()
    #epoch_edge_loss = AverageMeter()
    epoch_weighted_sum_loss = AverageMeter()
    start = time.time()

    for step, batch in enumerate(train_loader):
        cfg.global_step += 1

        # 调整学习率
        # cfg.lr = model.update_learning_rate_poly(cfg.global_step, cfg.lr, cfg.initial_lr, cfg.max_iter, cfg.power)

        model.set_input(batch, device)
        num = cfg.train_batch

        model.optimize_parameters()
        losses = model.get_current_losses()

        epoch_weighted_sum_loss.update(losses['sum'], num)
        epoch_seg_loss.update(losses['mask'], num)
        #epoch_edge_loss.update(losses['edge'], num)

        if cfg.global_step % snapshot == 0:
            logger.info('Epoch {} Step {}/{} '
                        'Loss {:.4f} ({:.4f}) '.format(epoch,
                                                       cfg.global_step,
                                                       len(train_loader),
                                                       epoch_weighted_sum_loss.val,
                                                       epoch_weighted_sum_loss.avg, )
                        )

    logger.info("[{}] Training - loss: {:.4e}".format(epoch, epoch_weighted_sum_loss.avg))

    elapsed = time.time() - start
    logger.info('Elapsed {:.2f}'.format(elapsed))

    if log_writer is not None:
        log_writer.add_scalar('Train/Loss', epoch_weighted_sum_loss.avg, epoch)
        log_writer.add_scalar('Train/seg_loss', epoch_seg_loss.avg, epoch)
        #log_writer.add_scalar('Train/edge_loss', epoch_edge_loss.avg, epoch)
        log_writer.add_scalar('Train/Time', elapsed, epoch)


def validate(epoch, model, device, val_loader, logger, log_writer, cfg):
    logger.info('Test {}'.format(epoch))

    model.eval()

    #val_edge_loss = AverageMeter()
    val_seg_loss = AverageMeter()
    val_weighted_loss = AverageMeter()
    start = time.time()

    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            model.set_input(batch, device)
            num = cfg.val_batch

            # hourglass model
            model.process()
            losses = model.get_current_losses()
            val_weighted_loss.update(losses['sum'], num)
            val_seg_loss.update(losses['mask'], num)
            #val_edge_loss.update(losses['edge'], num)

    logger.info("[{}] Val - loss: {:.4e}".format(epoch, val_weighted_loss.avg))

    elapsed = time.time() - start
    logger.info('Elapsed {:.2f}'.format(elapsed))

    if log_writer is not None:
        log_writer.add_scalar('Val/Loss', val_weighted_loss.avg, epoch)
        log_writer.add_scalar('Val/seg_loss', val_seg_loss.avg, epoch)
        #log_writer.add_scalar('Val/edge_loss', val_edge_loss.avg, epoch)
        log_writer.add_scalar('Val/Time', elapsed, epoch)

        if epoch == 0 or epoch % cfg.log_interval == 0 or epoch == cfg.max_epochs:
            model.show_eval_results(cfg.name, epoch, log_writer)

    return val_weighted_loss.avg
