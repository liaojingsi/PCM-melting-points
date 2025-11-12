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
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error,r2_score

#import _init_paths
from SOAP_model.config import config
from SOAP_model.config import update_config
from SOAP_model.config import save_config
from SOAP_model.core.loss import build_criterion
from SOAP_model.core.loss import build_criterion_reg
from SOAP_model.core.function import AverageMeter
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

# 线性函数反归一化
# data：需要归一化的数据
# gen_data：归一化之前的根数据用来确定最大值和最小值
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

def test_multi_with_reg_simpler(final_output_dir, epoch, begin_epoch, end_epoch, config_od, config_mol, val_loader, model, criterion_cls, criterion_reg, output_dir, tb_log_dir,
         writer_dict=None, distributed=False): #final_output_dir, epoch,
    batch_time = AverageMeter()
    losses_temp = AverageMeter()
    sysT = []
    predT = []
    targetT = []
    #top1 = AverageMeter()
    #top2 = AverageMeter()

    logging.info('=> switch to eval mode')
    model.eval()

    end = time.time()
    for i, (points, target_mol, target_od, target_temp) in enumerate(val_loader):

        points = points.transpose(2, 1)
        # compute output
        # points = points.cuda(non_blocking=True)
        # target_od = target_od.cuda(non_blocking=True)
        target_mol = target_mol.cuda(non_blocking=True)
        target_temp = target_temp.cuda(non_blocking=True)

        pred_temp, Tsys, output_feature_l2,correlation = model(points)

        loss_temp = criterion_reg(pred_temp, MaxMinNormalization((Tsys - target_temp.to(torch.float32)), -70, 0))#.sum()
        print(UnMaxMinNormalization(pred_temp, -70, 0) - (Tsys - target_temp))
        print(target_mol)
        print(loss_temp)

        # pred_temp = UnMaxMinNormalization(pred_temp, -70, 70)

        # sysT = np.append(sysT, Tsys.squeeze(-1).cpu().detach().numpy())
        # targetT = np.append(targetT, target_temp.cpu().detach().numpy())
        # predT = np.append(predT, pred_temp.cpu().detach().numpy())

        losses_temp.update(loss_temp.item(), points.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logging.info('=> synchronize...')
    comm.synchronize()

    # sysT = np.array(sysT).reshape(-1)
    # targetT = np.array(targetT).reshape(-1)
    # predT = np.array(predT).reshape(-1)

    # deltaT_pred = predT
    # deltaT_real = sysT - targetT

    loss_temp_avg= losses_temp.avg
    # r_2 = r2_score(deltaT_real, deltaT_pred)
    # loss_temp_avg = -r_2

    if comm.is_main_process():
        msg = '=> TEST:\t' \
              'Loss_temp {loss_temp_avg:.4f}\t'.format(
                loss_temp_avg=loss_temp_avg
            )
        logging.info(msg)

    if writer_dict and comm.is_main_process():
        global_steps = writer_dict['valid_global_steps']
        writer_dict['valid_global_steps'] = global_steps + 1

    logging.info('=> switch to train mode')
    model.train()

    return loss_temp_avg

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
            losses_temp = AverageMeter()
            #top1 = AverageMeter()
            #top2 = AverageMeter()

            logging.info('=> switch to train mode')
            model.train()

            end = time.time()

            for batch_id, (points, target_mol, target_od, target_temp) in enumerate(train_loader, 0):
                # measure data loading time
                data_time.update(time.time() - end)
                logging.info(f'batch_id {batch_id}\t'.format(batch_id=batch_id))
                print('batch_id:   ', batch_id)

                # compute output
                # points = points.data.numpy()
                # points[:, :, 2:5] = provider.rotate_point_cloud(points[:, :, 2:5])
                # points[:, :, 2:5] = provider.shift_point_cloud(points[:, :, 2:5])
                # points = torch.Tensor(points)
                points = points.transpose(2, 1)

                # points = points.cuda(non_blocking=True)
                # target_od = target_od.cuda(non_blocking=True)
                target_mol = target_mol.cuda(non_blocking=True)
                target_temp = target_temp.cuda(non_blocking=True)
                # torch.cuda.empty_cache()

                with autocast(enabled=config.AMP.ENABLED):
                    pred_temp, Tsys, output_feature_l2, correlation = model(points)
                    #torch.cuda.empty_cache()
                    loss3 = criterion3(pred_temp, MaxMinNormalization((Tsys - target_temp.to(torch.float32)), -70, 0))
                    # print(UnMaxMinNormalization(pred_temp, -70, 70) - (Tsys - target_temp))
                    # print(target_mol)
                    # print(loss3)
                    # print(pred_temp.shape)
                    # print(Tsys.shape)
                    # print(target_temp.shape)
                    # print(loss3)

                optimizer.zero_grad()
                is_second_order = hasattr(optimizer, 'is_second_order') \
                                  and optimizer.is_second_order


                #target = target.argmax(target, dim=1)
                #train_instance_acc = np.mean(mean_correct)
                #logging.info(f'instance_acc {train_instance_acc:.3f}\t'.format(train_instance_acc=train_instance_acc))
                #print(train_instance_acc)

                scaler.scale(loss3).backward(create_graph=is_second_order)

                if config.TRAIN.CLIP_GRAD_NORM > 0.0:
                    # Unscales the gradients of optimizer's assigned params in-place
                    scaler.unscale_(optimizer)

                    # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.TRAIN.CLIP_GRAD_NORM
                    )

                scaler.step(optimizer)
                scaler.update()
                # torch.cuda.empty_cache()

                losses_temp.update(loss3.item(), points.size(0))

                batch_time.update(time.time() - end)
                end = time.time()

                if batch_id % config.PRINT_FREQ == 0:
                    msg = '=> Epoch[{0}][{1}/{2}]: ' \
                          'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                          'Speed {speed:.1f} samples/s\t' \
                          'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                          'Loss_temp {loss_temp.val:.5f} ({loss_temp.avg:.5f})\t'.format(
                        epoch, batch_id, len(train_loader),
                        batch_time=batch_time,
                        speed=points.size(0) / batch_time.val,
                        data_time=data_time,
                        loss_temp=losses_temp)
                    logging.info(msg)

                torch.cuda.synchronize()
            #for over

            train_instance_acc1 = np.mean(mean_correct1)
            logging.info(f'instance_acc {train_instance_acc1:.3f}\t'.format(train_instance_acc1=train_instance_acc1))
            print(train_instance_acc1)

            if writer_dict and comm.is_main_process():
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_losslt', losses_temp.avg, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        logging.info(
            '=> {} train end, duration: {:.2f}s'
            .format(head, time.time()-start)
        )

        # evaluate on test set
        logging.info('=> {} test start'.format(head))
        val_start = time.time()

        if epoch >= config.TRAIN.EVAL_BEGIN_EPOCH:
            loss_temp = test_multi_with_reg_simpler(
                final_output_dir=final_output_dir, epoch=epoch,
                begin_epoch=config.TRAIN.BEGIN_EPOCH, end_epoch=config.TRAIN.END_EPOCH,
                config_od=config.MODEL.NUM_CLASSES_ORDER_DISORDER, config_mol=config.MODEL.NUM_CLASSES_MOL,
                val_loader=valid_loader, model=model,
                criterion_cls=criterion_eval, criterion_reg=criterion_eval_reg,
                output_dir=final_output_dir, tb_log_dir=tb_log_dir, writer_dict=writer_dict,
                distributed=args.distributed
            )

            print(loss_temp)

            best_model = loss_temp <= best_regloss_temp
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

