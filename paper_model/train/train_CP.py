import sys
sys.path.append('/data/jsliao')

import argparse
import logging
import os
import pprint
import time
from tqdm import tqdm

import torch
import torch.nn.parallel
import torch.optim
from torch.utils.collect_env import get_pretty_env_info
from torch.cuda.amp import autocast
from tensorboardX import SummaryWriter
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

#import _init_paths
from SOAP_model.config import config
from SOAP_model.config import update_config
from SOAP_model.config import save_config
from SOAP_model.core.loss import build_criterion
from SOAP_model.core.loss import build_criterion_reg
from SOAP_model.core.function import AverageMeter
from SOAP_model.core.evaluate import accuracy
from SOAP_model.core import provider
from SOAP_model.dataset import build_multi_dataloader_with_reg
from SOAP_model.model import build_model
from SOAP_model.optim import build_optimizer
from SOAP_model.scheduler import build_lr_scheduler
from SOAP_model.utils.comm import comm
from SOAP_model.utils.utils import create_logger
from SOAP_model.utils.utils import init_distributed
from SOAP_model.utils.utils import setup_cudnn
from SOAP_model.utils.utils import summary_model_on_master
from SOAP_model.utils.utils import resume_checkpoint_multi_with_reg
from SOAP_model.utils.utils import save_checkpoint_on_master_multi_with_reg
from SOAP_model.utils.utils import save_model_on_master

def MaxMinNormalization(data,min,max):
    data = (data - min) / (max - min)
    return data

def UnMaxMinNormalization(data,min,max):
    return data*(max-min) + min

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train classification network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    # distributed training
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--port", type=int, default=9000)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args

def _meter_reduce(meter):
    rank = comm.local_rank
    meter_sum = torch.FloatTensor([meter.sum]).cuda(rank)
    meter_count = torch.FloatTensor([meter.count]).cuda(rank)
    torch.distributed.reduce(meter_sum, 0)
    torch.distributed.reduce(meter_count, 0)
    meter_avg = meter_sum / meter_count

    return meter_avg.item()

def test_multi_with_mol_reg(final_output_dir, epoch, begin_epoch, end_epoch, config_od, config_mol, val_loader, model, criterion_cls, criterion_reg, output_dir, tb_log_dir,
         writer_dict=None, distributed=False): #final_output_dir, epoch,
    batch_time = AverageMeter()
    mean_correct_od = []
    mean_correct_mol = []
    class_od_acc = np.zeros((config_od, 3))
    class_mol_acc = np.zeros((config_mol, 3))
    losses_od = AverageMeter()
    losses_mol = AverageMeter()
    losses_temp = AverageMeter()
    losses_sum = AverageMeter()
    #top1 = AverageMeter()
    #top2 = AverageMeter()

    logging.info('=> switch to eval mode')
    model.eval()

    end = time.time()
    for i, (points, target_mol, target_od, target_temp) in enumerate(val_loader):

        points = points.transpose(2, 1)
        # compute output
        points = points.cuda(non_blocking=True)
        target_mol = target_mol.cuda(non_blocking=True)
        target_od = target_od.cuda(non_blocking=True)
        target_temp = target_temp.cuda(non_blocking=True)

        pred_od, pred_mol, pred_temp, Tsys, output_feature_l2, output_feature_l1_od, output_feature_l1_mol = model(points)
        pred_choice_od = pred_od.data.max(1)[1]
        pred_choice_mol = pred_mol.data.max(1)[1]

        loss_od = criterion_cls(pred_od, target_od.long())
        # print(target_od)
        loss_mol = criterion_cls(pred_mol, target_mol.long())
        loss_temp = criterion_reg(pred_temp, MaxMinNormalization((Tsys.squeeze(-1) - target_temp.to(torch.float32)), -70, 70))
        # print(target_temp)
        print(loss_temp)

        #loss_sum = loss_od + loss_temp
        loss_sum = loss_od + loss_mol + loss_temp

        losses_od.update(loss_od.item(), points.size(0))
        losses_mol.update(loss_mol.item(), points.size(0))
        losses_temp.update(loss_temp.item(), points.size(0))
        losses_sum.update(loss_sum.item(), points.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        for cat in np.unique(target_od.cpu()):
            classacc_od = pred_choice_od[target_od == cat].eq(target_od[target_od == cat].long().data).cpu().sum()
            class_od_acc[cat, 0] += classacc_od.item() / float(points[target_od == cat].size()[0])
            class_od_acc[cat, 1] += 1

        for cat in np.unique(target_mol.cpu()):
            classacc_mol = pred_choice_mol[target_mol == cat].eq(target_mol[target_mol == cat].long().data).cpu().sum()
            class_mol_acc[cat, 0] += classacc_mol.item() / float(points[target_mol == cat].size()[0])
            class_mol_acc[cat, 1] += 1

        correct_od = pred_choice_od.eq(target_od.long().data).cpu().sum()
        mean_correct_od.append(correct_od.item() / float(points.size()[0]))

        correct_mol = pred_choice_mol.eq(target_mol.long().data).cpu().sum()
        mean_correct_mol.append(correct_mol.item() / float(points.size()[0]))

    class_od_acc[:, 2] = class_od_acc[:, 0] / class_od_acc[:, 1]
    class_od_acc = np.mean(class_od_acc[:, 2])
    print('class_od_acc', class_od_acc)
    instance_od_acc = np.mean(mean_correct_od)
    logging.info(f'instance_od_acc {instance_od_acc:.3f}\t'.format(instance_od_acc=instance_od_acc))
    print('instance_od_acc', instance_od_acc)

    class_mol_acc[:, 2] = class_mol_acc[:, 0] / class_mol_acc[:, 1]
    class_mol_acc = np.mean(class_mol_acc[:, 2])
    print('class_mol_acc', class_mol_acc)
    instance_mol_acc = np.mean(mean_correct_mol)
    logging.info(f'instance_mol_acc {instance_mol_acc:.3f}\t'.format(instance_mol_acc=instance_mol_acc))
    print('instance_mol_acc', instance_mol_acc)

    logging.info('=> synchronize...')
    comm.synchronize()

    loss_od_avg, loss_mol_avg, loss_temp_avg, loss_sum_avg= map(
        _meter_reduce if distributed else lambda x: x.avg,
        [losses_od, losses_mol, losses_temp, losses_sum]
    )

    if comm.is_main_process():
        msg = '=> TEST:\t' \
            'Loss_od {loss_od_avg:.4f}\t' \
            'Loss_mol {loss_mol_avg:.4f}\t' \
              'Loss_temp {loss_temp_avg:.4f}\t' \
              'Loss_sum {loss_sum_avg:.4f}\t'.format(
                loss_od_avg=loss_od_avg, loss_mol_avg=loss_mol_avg,
                loss_temp_avg=loss_temp_avg, loss_sum_avg=loss_sum_avg
            )
        logging.info(msg)

    if writer_dict and comm.is_main_process():
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_od_acc', class_od_acc, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    logging.info('=> switch to train mode')
    model.train()

    return instance_od_acc, instance_mol_acc, loss_temp_avg

def main():
    args = parse_args()

    init_distributed(args)
    setup_cudnn(config)

    update_config(config, args)
    final_output_dir = create_logger(config, args.cfg, 'train')
    tb_log_dir = final_output_dir

    if comm.is_main_process():
        logging.info("=> collecting env info (might take some time)")
        logging.info("\n" + get_pretty_env_info())
        logging.info(pprint.pformat(args))
        logging.info(config)
        logging.info("=> using {} GPUs".format(args.num_gpus))

        output_config_path = os.path.join(final_output_dir, 'config.yaml')
        logging.info("=> saving config into: {}".format(output_config_path))
        save_config(config, output_config_path)

    model = build_model(config)
    model.to(torch.device('cuda'))

    # copy model file
    summary_model_on_master(model, config, final_output_dir, True)

    writer_dict = {
        'writer': SummaryWriter(logdir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    best_perf_od = 0.0
    best_perf_mol = 0.0
    best_regloss_temp= 1e6
    best_model = True
    begin_epoch = config.TRAIN.BEGIN_EPOCH
    optimizer = build_optimizer(config, model)

    best_perf_od, best_perf_mol, best_regloss_temp, begin_epoch = resume_checkpoint_multi_with_reg(
        model, optimizer, config, final_output_dir, True
    )

    print('get_train_dataset')
    train_loader = build_multi_dataloader_with_reg(config, 'train', True, args.distributed)
    print('get_test_dataset')
    valid_loader = build_multi_dataloader_with_reg(config, 'test', False, args.distributed)
    print('get_dataset_done')

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
        )

    criterion1 = build_criterion(config)
    criterion2 = build_criterion(config)
    criterion3 = build_criterion_reg()
    criterion4 = build_criterion_reg()
    criterion1.cuda()
    criterion2.cuda()
    criterion3.cuda()
    criterion4.cuda()
    criterion_eval = build_criterion(config, train=False)
    criterion_eval_reg = build_criterion_reg()
    criterion_eval.cuda()
    criterion_eval_reg.cuda()

    lr_scheduler = build_lr_scheduler(config, optimizer, begin_epoch)

    scaler = torch.cuda.amp.GradScaler(enabled=config.AMP.ENABLED)

    logging.info('=> start training')

    for epoch in range(begin_epoch, config.TRAIN.END_EPOCH):
        head = 'Epoch[{}]:'.format(epoch)
        logging.info('=> {} epoch start'.format(head))
        mean_correct1 = []
        mean_correct2 = []

        start = time.time()
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        logging.info('=> {} train start'.format(head))

        with torch.autograd.set_detect_anomaly(config.TRAIN.DETECT_ANOMALY):
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses1 = AverageMeter()
            losses2 = AverageMeter()
            losses_temp = AverageMeter()
            losses = AverageMeter()
            #top1 = AverageMeter()
            #top2 = AverageMeter()

            logging.info('=> switch to train mode')
            model.train()

            end = time.time()

            # for batch_id, points, target_mol, target_od, target_temp in enumerate(train_loader, 0):
            #     print(batch_id, points, target_mol, target_od, target_temp)

            for batch_id, (points, target_mol, target_od, target_temp) in enumerate(train_loader, 0):
                # measure data loading time
                data_time.update(time.time() - end)
                logging.info(f'batch_id {batch_id}\t'.format(batch_id=batch_id))
                print('batch_id:   ', batch_id)

                # compute output
                # points = points.data.numpy()
                # points[:, :, 3:6] = provider.rotate_point_cloud(points[:, :, 3:6])
                # points[:, :, 3:6] = provider.shift_point_cloud(points[:, :, 3:6])
                # points = torch.Tensor(points)
                points = points.transpose(2, 1)

                points = points.cuda(non_blocking=True)
                target_mol = target_mol.cuda(non_blocking=True)
                target_od = target_od.cuda(non_blocking=True)
                target_temp = target_temp.cuda(non_blocking=True)

                with autocast(enabled=config.AMP.ENABLED):
                    pred_order, pred_mol, pred_temp, Tsys, output_feature_l2, output_feature_l1_od, output_feature_l1_mol = model(points)
                    loss1 = criterion1(pred_order, target_od.long())
                    #print(loss1)
                    loss2 = criterion2(pred_mol, target_mol.long())
                    # print(loss2)
                    loss3 = criterion3(pred_temp, MaxMinNormalization((Tsys.squeeze(-1) - target_temp.to(torch.float32)), -70, 70))
                    #print(target_temp)
                    # print(loss3)

                pred_choice1 = pred_order.data.max(1)[1]

                correct1 = pred_choice1.eq(target_od.long().data).cpu().sum()
                mean_correct1.append(correct1.item() / float(points.size()[0]))

                pred_choice2 = pred_mol.data.max(1)[1]

                correct2 = pred_choice2.eq(target_mol.long().data).cpu().sum()
                mean_correct2.append(correct2.item() / float(points.size()[0]))

                loss_sum = loss1 + loss2 + loss3

                optimizer.zero_grad()
                is_second_order = hasattr(optimizer, 'is_second_order') \
                                  and optimizer.is_second_order


                #target = target.argmax(target, dim=1)
                #train_instance_acc = np.mean(mean_correct)
                #logging.info(f'instance_acc {train_instance_acc:.3f}\t'.format(train_instance_acc=train_instance_acc))
                #print(train_instance_acc)

                scaler.scale(loss_sum).backward(create_graph=is_second_order)

                if config.TRAIN.CLIP_GRAD_NORM > 0.0:
                    # Unscales the gradients of optimizer's assigned params in-place
                    scaler.unscale_(optimizer)

                    # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.TRAIN.CLIP_GRAD_NORM
                    )

                scaler.step(optimizer)
                scaler.update()

                losses1.update(loss1.item(), points.size(0))
                losses2.update(loss2.item(), points.size(0))
                losses_temp.update(loss3.item(), points.size(0))
                losses.update(loss_sum.item(), points.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if batch_id % config.PRINT_FREQ == 0:
                    msg = '=> Epoch[{0}][{1}/{2}]: ' \
                          'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                          'Speed {speed:.1f} samples/s\t' \
                          'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                          'Loss_od {loss_od.val:.5f} ({loss_od.avg:.5f})\t' \
                          'Loss_mol {loss_mol.val:.5f} ({loss_mol.avg:.5f})\t' \
                          'Loss_temp {loss_temp.val:.5f} ({loss_temp.avg:.5f})\t' \
                          'Loss_sum {loss_sum.val:.5f} ({loss_sum.avg:.5f})\t'.format(
                        epoch, batch_id, len(train_loader),
                        batch_time=batch_time,
                        speed=points.size(0) / batch_time.val,
                        data_time=data_time, loss_od=losses1, loss_mol=losses2,
                        loss_temp=losses_temp, loss_sum=losses)
                    logging.info(msg)

                torch.cuda.synchronize()
            #for over

            train_instance_acc1 = np.mean(mean_correct1)
            train_instance_acc2 = np.mean(mean_correct2)
            logging.info(f'instance_acc {train_instance_acc1:.3f}\t'.format(train_instance_acc1=train_instance_acc1))
            logging.info(f'instance_acc {train_instance_acc2:.3f}\t'.format(train_instance_acc2=train_instance_acc2))
            print(train_instance_acc1)
            print(train_instance_acc2)

            if writer_dict and comm.is_main_process():
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_lossod', losses1.avg, global_steps)
                writer.add_scalar('train_lossmol', losses2.avg, global_steps)
                writer.add_scalar('train_losslt', losses_temp.avg, global_steps)
                writer.add_scalar('train_loss_sum', losses.avg, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        logging.info(
            '=> {} train end, duration: {:.2f}s'
            .format(head, time.time()-start)
        )

        # evaluate on test set
        logging.info('=> {} test start'.format(head))
        val_start = time.time()

        if epoch >= config.TRAIN.EVAL_BEGIN_EPOCH:
            perf_od, perf_mol, loss_temp = test_multi_with_mol_reg(
                final_output_dir=final_output_dir, epoch=epoch,
                begin_epoch=config.TRAIN.BEGIN_EPOCH, end_epoch=config.TRAIN.END_EPOCH,
                config_od=config.MODEL.NUM_CLASSES_ORDER_DISORDER, config_mol=config.MODEL.NUM_CLASSES_MOL,
                val_loader=valid_loader, model=model,
                criterion_cls=criterion_eval, criterion_reg=criterion_eval_reg,
                output_dir=final_output_dir, tb_log_dir=tb_log_dir, writer_dict=writer_dict,
                distributed=args.distributed
            )

            best_model = loss_temp <= best_regloss_temp
            best_perf_od = perf_od if best_model else best_perf_od
            best_perf_mol = perf_mol if best_model else best_perf_mol
            best_regloss_temp = loss_temp if best_model else best_regloss_temp

        logging.info(
            '=> {} validate end, duration: {:.2f}s'
            .format(head, time.time() - val_start)
        )

        lr_scheduler.step(epoch=epoch+1)
        if config.TRAIN.LR_SCHEDULER.METHOD == 'timm':
            lr = lr_scheduler.get_epoch_values(epoch+1)[0]
        else:
            lr = lr_scheduler.get_last_lr()[0]
        logging.info(f'=> lr: {lr}')

        save_checkpoint_on_master_multi_with_reg(
            model=model,
            distributed=args.distributed,
            model_name=config.MODEL.NAME,
            optimizer=optimizer,
            output_dir=final_output_dir,
            in_epoch=True,
            epoch_or_step=epoch,
            best_perf_od=best_perf_od,
            best_perf_mol=best_perf_mol,
            best_regloss_temp=best_regloss_temp
        )

        if best_model and comm.is_main_process():
            save_model_on_master(
                model, args.distributed, final_output_dir, 'model_best.pth'
            )

        if config.TRAIN.SAVE_ALL_MODELS and comm.is_main_process():
            save_model_on_master(
                model, args.distributed, final_output_dir, f'model_{epoch}.pth'
            )

        logging.info(
            '=> {} epoch end, duration : {:.2f}s'
            .format(head, time.time()-start)
        )

        save_model_on_master(
            model, args.distributed, final_output_dir, 'final_state.pth'
        )

        #if config.SWA.ENABLED and comm.is_main_process():
        #    save_model_on_master(
        #        args.distributed, final_output_dir, 'swa_state.pth'
        #    )

        writer_dict['writer'].close()
        logging.info('=> finish training')




if __name__ == '__main__':
    main()

