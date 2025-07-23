import sys
sys.path.append('../..')

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
from paper_model.config import config
from paper_model.config import update_config
from paper_model.config import save_config
from paper_model.core.loss import build_criterion
from paper_model.core.loss import build_criterion_reg
from paper_model.core.function import AverageMeter
from paper_model.core.evaluate import accuracy
from paper_model.core import provider
from paper_model.dataset import build_multi_dataloader_with_reg
from paper_model.model import build_model
from paper_model.optim import build_optimizer
from paper_model.scheduler import build_lr_scheduler
from paper_model.utils.comm import comm
from paper_model.utils.utils import create_logger
from paper_model.utils.utils import init_distributed
from paper_model.utils.utils import setup_cudnn
from paper_model.utils.utils import summary_model_on_master
from paper_model.utils.utils import resume_checkpoint_multi_with_reg
from paper_model.utils.utils import save_checkpoint_on_master_multi_with_reg
from paper_model.utils.utils import save_model_on_master

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error,r2_score

def parse_args():
    parser = argparse.ArgumentParser(
        description='Test classification network')

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

def MaxMinNormalization(data,min,max):
    data = (data - min) / (max - min)
    return data

def UnMaxMinNormalization(data,min,max):
    return data*(max-min) + min

@torch.no_grad()
def test_multi_with_reg_simpler_test(final_output_dir, val_loader, model, criterion_reg,
         writer_dict=None): #final_output_dir, epoch,
    batch_time = AverageMeter()
    losses_temp = AverageMeter()


    logging.info('=> switch to eval mode')
    model.eval()

    sysT = []
    predT = []
    targetT = []

    end = time.time()
    for i, (points, target_mol, target_od, target_temp) in enumerate(val_loader):

        points = points.transpose(2, 1)
        # compute output
        points = points.cuda(non_blocking=True)
        target_temp = target_temp.cuda(non_blocking=True)

        pred_temp, Tsys, output_feature_l2,  correlation = model(points)

        loss_temp = criterion_reg(pred_temp, MaxMinNormalization((Tsys - target_temp.to(torch.float32)), -70, 0))#.sum()
        # print(pred_temp)
        print(loss_temp)

        pred_temp = UnMaxMinNormalization(pred_temp, -70, 0)

        losses_temp.update(loss_temp.item(), points.size(0))

        sysT = np.append(sysT, Tsys.squeeze(-1).cpu().detach().numpy())
        targetT = np.append(targetT, target_temp.cpu().detach().numpy())
        predT = np.append(predT, pred_temp.cpu().detach().numpy())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logging.info('=> synchronize...')
    comm.synchronize()

    print(np.mean(predT))

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.serif'] = ['Arial']

    loss_temp_avg= losses_temp.avg

    sysT = np.array(sysT).reshape(-1)
    targetT = np.array(targetT).reshape(-1)
    predT = np.array(predT).reshape(-1)

    deltaT_pred = predT
    deltaT_real = sysT - targetT

    mae = mean_absolute_error(deltaT_real[:800], deltaT_pred[:800])
    rmse = np.sqrt(mean_squared_error(deltaT_real[:800], deltaT_pred[:800]))
    r_2 = r2_score(deltaT_real[:800], deltaT_pred[:800])

    x45 = np.linspace(-70.0, 0, 100)
    y45=x45

    plt.figure(figsize=(2, 2))
    plt.plot(x45, y45, c='orangered', linewidth=2, alpha=0.8)
    plt.scatter(deltaT_real, deltaT_pred, s=3, c='dodgerblue', edgecolors='blue', linewidths=0.3)
    plt.text(-68, -4, s=f"$R^2$={round(r_2, 2)}", fontsize=5)
    plt.text(-68, -7, 'RMSE={:.2f}'.format(rmse), fontsize=5)
    plt.text(-68, -10, 'MAE={:.2f}'.format(mae), fontsize=5)
    plt.xlim([-70.0, 0])
    plt.xticks([-60,-40,-20,0])
    plt.ylim([-70.0, 0])
    plt.yticks([-60,-40,-20,0])
    plt.tick_params(axis='both', length=2, pad=2, labelsize=5)
    # plt.axis('square')
    plt.xlabel('MD temperature (K)', fontsize=5, va='center')
    plt.ylabel('Predicted temperature (K)', fontsize=5, va='center')
    plt.tight_layout()

    plt.savefig(os.path.join(final_output_dir, 'test_Epoch_rmse_solid.png'), dpi=600)

    plt.close("all")

    if comm.is_main_process():
        msg = '=> TEST:\t' \
              'Loss_temp {loss_temp_avg:.4f}\t'.format(
                loss_temp_avg=loss_temp_avg
            )
        logging.info(msg)

    if writer_dict and comm.is_main_process():
        writer = writer_dict['writer']
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
    final_output_dir = create_logger(config, args.cfg, 'test')
    tb_log_dir = final_output_dir

    if comm.is_main_process():
        logging.info("=> collecting env info (might take some time)")
        logging.info("\n" + get_pretty_env_info())
        logging.info(pprint.pformat(args))
        logging.info(config)
        logging.info("=> using {} GPUs".format(args.num_gpus))

        output_config_path = os.path.join(final_output_dir, 'config.yaml')
        logging.info("=> saving config into: {}".format(output_config_path))

    model = build_model(config)
    model.to(torch.device('cuda'))

    model_file = config.TEST.MODEL_FILE if config.TEST.MODEL_FILE \
        else os.path.join(final_output_dir, 'model_best.pth')
    logging.info('=> load model file: {}'.format(model_file))
    ext = model_file.split('.')[-1]
    if ext == 'pth':
        state_dict = torch.load(model_file, map_location="cpu")
    else:
        raise ValueError("Unknown model file")

    model.load_state_dict(state_dict, strict=False)
    model.to(torch.device('cuda'))

    writer_dict = {
        'writer': SummaryWriter(logdir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    summary_model_on_master(model, config, final_output_dir, False)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
        )

    # define loss function (criterion) and optimizer
    criterion_eval_reg = build_criterion_reg()
    criterion_eval_reg.cuda()

    valid_loader = build_multi_dataloader_with_reg(config, 'test', False, args.distributed)

    logging.info('=> start testing')
    start = time.time()
    test_multi_with_reg_simpler_test(
        final_output_dir=final_output_dir,
        val_loader=valid_loader, model=model, criterion_reg=criterion_eval_reg, writer_dict=writer_dict
    )
    logging.info('=> test duration time: {:.2f}s'.format(time.time()-start))

    writer_dict['writer'].close()
    logging.info('=> finish testing')


if __name__ == '__main__':
    main()




