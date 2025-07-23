import logging
import time
import torch
import os

from timm.data import Mixup
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from torch.cuda.amp import autocast
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error,r2_score

from .evaluate import accuracy
from paper_model.utils.comm import comm

def MaxMinNormalization(data,min,max):
    data = (data - min) / (max - min)
    return data

def UnMaxMinNormalization(data,min,max):
    return data*(max-min) + min


def plt_color(labels):
    col=[]
    label = []
    color_list = ['r',
                  'chocolate',
                  'darkorange',
                  'gold',
                  'forestgreen',
                  'springgreen',
                  'darkcyan',
                  'blue',
                  'darkviolet',
                  'deeppink']
    label_list = ['1188514',
                               '1286153',
                               '1142500',
                               '248562',
                               '1034001',
                               '1450800',
                               '1249410',
                               '1298203',
                               '1432562',
                               '2150172']
    for i in range(0, len(labels)):
        col.append(color_list[labels[i]])
        label.append(label_list[labels[i]])
    return col, label

def train_one_epoch(config, train_loader, model, criterion, optimizer, epoch,
                    output_dir, tb_log_dir, writer_dict, scaler=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    logging.info('=> switch to train mode')
    model.train()

    aug = config.AUG
    mixup_fn = Mixup(
        mixup_alpha=aug.MIXUP, cutmix_alpha=aug.MIXCUT,
        cutmix_minmax=aug.MIXCUT_MINMAX if aug.MIXCUT_MINMAX else None,
        prob=aug.MIXUP_PROB, switch_prob=aug.MIXUP_SWITCH_PROB,
        mode=aug.MIXUP_MODE, label_smoothing=config.LOSS.LABEL_SMOOTHING,
        num_classes=config.MODEL.NUM_CLASSES
    ) if aug.MIXUP_PROB > 0.0 else None
    end = time.time()
    for i, (x, y) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        if mixup_fn:
            x, y = mixup_fn(x, y)

        with autocast(enabled=config.AMP.ENABLED):
            if config.AMP.ENABLED and config.AMP.MEMORY_FORMAT == 'nwhc':
                x = x.contiguous(memory_format=torch.channels_last)
                y = y.contiguous(memory_format=torch.channels_last)

            outputs = model(x)
            loss = criterion(outputs, y)

        # compute gradient and do update step
        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') \
            and optimizer.is_second_order

        scaler.scale(loss).backward(create_graph=is_second_order)

        if config.TRAIN.CLIP_GRAD_NORM > 0.0:
            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)

            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.TRAIN.CLIP_GRAD_NORM
            )

        scaler.step(optimizer)
        scaler.update()
        # measure accuracy and record loss
        losses.update(loss.item(), x.size(0))

        if mixup_fn:
            y = torch.argmax(y, dim=1)
        prec1, prec5 = accuracy(outputs, y, (1, 5))

        top1.update(prec1, x.size(0))
        top5.update(prec5, x.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = '=> Epoch[{0}][{1}/{2}]: ' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy@1 {top1.val:.3f} ({top1.avg:.3f})\t' \
                  'Accuracy@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                      epoch, i, len(train_loader),
                      batch_time=batch_time,
                      speed=x.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, top1=top1, top5=top5)
            logging.info(msg)

        torch.cuda.synchronize()

    if writer_dict and comm.is_main_process():
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        writer.add_scalar('train_loss', losses.avg, global_steps)
        writer.add_scalar('train_top1', top1.avg, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1


@torch.no_grad()
def test(config, val_loader, model, criterion, output_dir, tb_log_dir,
         writer_dict=None, distributed=False): #final_output_dir, epoch,
    batch_time = AverageMeter()
    mean_correct = []
    class_acc = np.zeros((config, 3))
    #top1 = AverageMeter()
    #top2 = AverageMeter()

    logging.info('=> switch to eval mode')
    model.eval()

    end = time.time()
    for i, (x, y) in enumerate(val_loader):

        x = x.transpose(2, 1)
        # compute output
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        outputs, output_feature_l2, output_feature_l1 = model(x)
        pred_choice = outputs.data.max(1)[1]

        #conv5_output = output_feature_l1.cpu().detach().numpy()#.reshape(32, 1125*200)#.reshape(32, 3*200)
       # pca = PCA(n_components=0.95)
        #pca_result = pca.fit_transform(conv5_output)
        #N_components = pca.n_components_
        # component_colors = {}
        # list = np.arange(N_components)
        # step_size = (256 ** 3) // len(list)
        # for i, component in enumerate(list):
        #    component_colors[component] = '#{}'.format(hex(step_size * i)[2:])
        # colors = [component_colors[component] for component in list]
        # print(len(pca_result[:, 0]))
        # print(len(pca_result[:, 1]))
        # print(len(colors))
        # df1 = pd.DataFrame(dict(pca_result0=pca_result[:, 0], pca_result1=pca_result[:, 1], colors=colors))
        #print(N_components)
        #plt.figure(figsize=(10, 10))
       # plot = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=pred_choice.cpu().detach().numpy()) #c=pred_choice.cpu().detach().numpy()
        #plt.legend(handles=plot.legend_elements()[0])
        # if epoch>17:
        # plt.show()
        #plt.savefig(os.path.join(final_output_dir, 'test_Epoch[{0}]_{1}.png'.format(epoch, i)), dpi=300)
        #plt.close("all")

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        for cat in np.unique(y.cpu()):
            classacc = pred_choice[y == cat].eq(y[y == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(x[y == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(y.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(x.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    print('class_acc', class_acc)
    instance_acc = np.mean(mean_correct)
    logging.info(f'instance_acc {instance_acc:.3f}\t'.format(instance_acc=instance_acc))
    print('instance_acc', instance_acc)

    logging.info('=> synchronize...')
    comm.synchronize()

    if writer_dict and comm.is_main_process():
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_acc', class_acc, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    logging.info('=> switch to train mode')
    model.train()

    return instance_acc

@torch.no_grad()
def test_multi(final_output_dir, epoch, begin_epoch, end_epoch, config_od, config_mol, val_loader, model, criterion, output_dir, tb_log_dir,
         writer_dict=None, distributed=False): #final_output_dir, epoch,
    batch_time = AverageMeter()
    mean_correct_od = []
    mean_correct_mol = []
    class_od_acc = np.zeros((config_od, 3))
    class_mol_acc = np.zeros((config_mol, 3))
    losses_od = AverageMeter()
    losses_mol = AverageMeter()
    losses_sum = AverageMeter()
    #top1 = AverageMeter()
    #top2 = AverageMeter()

    logging.info('=> switch to eval mode')
    model.eval()

    end = time.time()
    for i, (points, target_mol, target_od) in enumerate(val_loader):

        points = points.transpose(2, 1)
        # compute output
        points = points.cuda(non_blocking=True)
        target_mol = target_mol.cuda(non_blocking=True)
        target_od = target_od.cuda(non_blocking=True)

        pred_od, pred_mol, output_feature_l2, output_feature_l1 = model(points)
        pred_choice_od = pred_od.data.max(1)[1]
        pred_choice_mol = pred_mol.data.max(1)[1]

        loss_od = criterion(pred_od, target_od.long())
        loss_mol = criterion(pred_mol, target_mol.long())
        loss_sum = loss_od + loss_mol

        losses_od.update(loss_od.item(), points.size(0))
        losses_mol.update(loss_mol.item(), points.size(0))
        losses_sum.update(loss_sum.item(), points.size(0))

        if epoch == end_epoch-1:
            conv5_output = output_feature_l1.cpu().detach().numpy()  # .reshape(32, 1125*200)#.reshape(32, 3*200)
            distortion_output = output_feature_l2.cpu().detach().numpy().reshape(128, 512 * 1000)  # .reshape(32, 3*200)
            tsne2d = TSNE(n_components=2, init='pca', random_state=0)
            tsne_result = tsne2d.fit_transform(conv5_output)
            tsne_result_d = tsne2d.fit_transform(distortion_output)
            # N_components = pca.n_components_
            # component_colors = {}
            # list = np.arange(N_components)
            # step_size = (256 ** 3) // len(list)
            # for i, component in enumerate(list):
            #    component_colors[component] = '#{}'.format(hex(step_size * i)[2:])
            # colors = [component_colors[component] for component in list]
            # print(len(pca_result[:, 0]))
            # print(len(pca_result[:, 1]))
            # print(len(colors))
            # df1 = pd.DataFrame(dict(pca_result0=pca_result[:, 0], pca_result1=pca_result[:, 1], colors=colors))
            # print(N_components)
            plt.figure(figsize=(26, 6))
            plot = plt.subplot(141)
            plt1 = plot.scatter(tsne_result[:, 0], tsne_result[:, 1],
                                c=target_mol.cpu().detach().numpy(), cmap='rainbow')  # c=pred_choice.cpu().detach().numpy()
            plt.title('mlp_mol_test', x=0.5, y=-0.1)
            plt.legend(handles=plt1.legend_elements()[0],
                       loc='best',
                       labels=['1188514',
                               '1286153',
                               '1142500',
                               '248562',
                               '1034001',
                               '1450800',
                               '1249410',
                               '1298203',
                               '1432562',
                               '2150172'])  # handles=plot.legend_elements()[0],labels=['1188514', '1286153']
            plot = plt.subplot(142)
            plt2 = plot.scatter(tsne_result[:, 0], tsne_result[:, 1],
                                c=target_od.cpu().detach().numpy())  # c=pred_choice.cpu().detach().numpy()
            plt.title('mlp_od_test', x=0.5, y=-0.1)
            plt.legend(handles=plt2.legend_elements()[0],
                       loc='best',
                       labels=['order', 'disorder'])  # handles=plot.legend_elements()[0],labels=['order', 'disorder']
            plot = plt.subplot(143)
            plt3 = plot.scatter(tsne_result_d[:, 0], tsne_result_d[:, 1],
                                c=target_mol.cpu().detach().numpy(), cmap='rainbow')  # c=pred_choice.cpu().detach().numpy()
            plt.title('conv_mol_test', x=0.5, y=-0.1)
            plt.legend(handles=plt3.legend_elements()[0], loc='best',
                       labels=['1188514',
                               '1286153',
                               '1142500',
                               '248562',
                               '1034001',
                               '1450800',
                               '1249410',
                               '1298203',
                               '1432562',
                               '2150172'])  # handles=plot.legend_elements()[0],labels=['1188514', '1286153']
            plot = plt.subplot(144)
            plt4 = plot.scatter(tsne_result_d[:, 0], tsne_result_d[:, 1],
                                c=target_od.cpu().detach().numpy())  # c=pred_choice.cpu().detach().numpy()
            plt.title('conv_od_test', x=0.5, y=-0.1)
            plt.legend(handles=plt4.legend_elements()[0], loc='best', labels=['order',
                                                                              'disorder'])  # handles=plot.legend_elements()[0],labels=['order', 'disorder']
            # if epoch>17:
            # plt.show()
            plt.savefig(os.path.join(final_output_dir, 'test_Epoch[{0}]_{1}.png'.format(epoch, i)), dpi=720)
            plt.close("all")


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

    loss_od_avg, loss_mol_avg ,loss_sum_avg= map(
        _meter_reduce if distributed else lambda x: x.avg,
        [losses_od, losses_mol, losses_sum]
    )

    if comm.is_main_process():
        msg = '=> TEST:\t' \
            'Loss_od {loss_od_avg:.4f}\t' \
            'Loss_mol {loss_mol_avg:.4f}\t'\
            'Loss_sum {loss_sum_avg:.4f}\t'.format(
                loss_od_avg=loss_od_avg, loss_mol_avg=loss_mol_avg, loss_sum_avg=loss_sum_avg
            )
        logging.info(msg)

    if writer_dict and comm.is_main_process():
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_od_acc', class_od_acc, global_steps)
        writer.add_scalar('valid_mol_acc', class_mol_acc, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    logging.info('=> switch to train mode')
    model.train()

    return instance_od_acc, instance_mol_acc


@torch.no_grad()
def test_multi_from_sa3(final_output_dir, epoch, begin_epoch, end_epoch, config_od, config_mol, val_loader, model, criterion, output_dir, tb_log_dir,
         writer_dict=None, distributed=False): #final_output_dir, epoch,
    batch_time = AverageMeter()
    mean_correct_od = []
    mean_correct_mol = []
    class_od_acc = np.zeros((config_od, 3))
    class_mol_acc = np.zeros((config_mol, 3))
    losses_od = AverageMeter()
    losses_mol = AverageMeter()
    losses_sum = AverageMeter()
    #top1 = AverageMeter()
    #top2 = AverageMeter()

    logging.info('=> switch to eval mode')
    model.eval()

    end = time.time()
    for i, (points, target_mol, target_od) in enumerate(val_loader):

        points = points.transpose(2, 1)
        # compute output
        points = points.cuda(non_blocking=True)
        target_mol = target_mol.cuda(non_blocking=True)
        target_od = target_od.cuda(non_blocking=True)

        pred_od, pred_mol, output_feature_l2, output_feature_l1_od, output_feature_l1_mol = model(points)
        pred_choice_od = pred_od.data.max(1)[1]
        pred_choice_mol = pred_mol.data.max(1)[1]

        loss_od = criterion(pred_od, target_od.long())
        loss_mol = criterion(pred_mol, target_mol.long())
        loss_sum = loss_od + loss_mol

        losses_od.update(loss_od.item(), points.size(0))
        losses_mol.update(loss_mol.item(), points.size(0))
        losses_sum.update(loss_sum.item(), points.size(0))

        if epoch == end_epoch-1:
            output_od = output_feature_l1_od.cpu().detach().numpy()  # .reshape(32, 1125*200)#.reshape(32, 3*200)
            output_mol = output_feature_l1_mol.cpu().detach().numpy()  # .reshape(32, 1125*200)#.reshape(32, 3*200)
            distortion_output = output_feature_l2.cpu().detach().numpy().reshape(128, 512 * 1000)  # .reshape(32, 3*200)
            tsne2d = TSNE(n_components=2, init='pca', random_state=0)
            tsne_result_od = tsne2d.fit_transform(output_od)
            tsne_result_mol = tsne2d.fit_transform(output_mol)
            tsne_result_d = tsne2d.fit_transform(distortion_output)
            # N_components = pca.n_components_
            # component_colors = {}
            # list = np.arange(N_components)
            # step_size = (256 ** 3) // len(list)
            # for i, component in enumerate(list):
            #    component_colors[component] = '#{}'.format(hex(step_size * i)[2:])
            # colors = [component_colors[component] for component in list]
            # print(len(pca_result[:, 0]))
            # print(len(pca_result[:, 1]))
            # print(len(colors))
            # df1 = pd.DataFrame(dict(pca_result0=pca_result[:, 0], pca_result1=pca_result[:, 1], colors=colors))
            # print(N_components)
            plt.figure(figsize=(26, 6))
            plot = plt.subplot(141)
            plt1 = plot.scatter(tsne_result_mol[:, 0], tsne_result_mol[:, 1],
                                c=target_mol.cpu().detach().numpy(), cmap='rainbow')  # c=pred_choice.cpu().detach().numpy()
            plt.title('mlp_mol_test', x=0.5, y=-0.1)
            plt.legend(handles=plt1.legend_elements()[0],
                       loc='best',
                       labels=['1188514',
                               '1286153',
                               '1142500',
                               '248562',
                               '1034001',
                               '1450800',
                               '1249410',
                               '1298203',
                               '1432562',
                               '2150172'])  # handles=plot.legend_elements()[0],labels=['1188514', '1286153']
            plot = plt.subplot(142)
            plt2 = plot.scatter(tsne_result_od[:, 0], tsne_result_od[:, 1],
                                c=target_od.cpu().detach().numpy())  # c=pred_choice.cpu().detach().numpy()
            plt.title('mlp_od_test', x=0.5, y=-0.1)
            plt.legend(handles=plt2.legend_elements()[0],
                       loc='best',
                       labels=['order', 'disorder'])  # handles=plot.legend_elements()[0],labels=['order', 'disorder']
            plot = plt.subplot(143)
            plt3 = plot.scatter(tsne_result_d[:, 0], tsne_result_d[:, 1],
                                c=target_mol.cpu().detach().numpy(), cmap='rainbow')  # c=pred_choice.cpu().detach().numpy()
            plt.title('conv_mol_test', x=0.5, y=-0.1)
            plt.legend(handles=plt3.legend_elements()[0], loc='best',labels=['1188514',
                                                                  '1286153',
                                                                  '1142500',
                                                                  '248562',
                                                                  '1034001',
                                                                  '1450800',
                                                                  '1249410',
                                                                  '1298203',
                                                                  '1432562',
                                                                  '2150172'])  # handles=plot.legend_elements()[0],labels=['1188514', '1286153']
            plot = plt.subplot(144)
            plt4 = plot.scatter(tsne_result_d[:, 0], tsne_result_d[:, 1],
                                c=target_od.cpu().detach().numpy())  # c=pred_choice.cpu().detach().numpy()
            plt.title('mlp_od_test', x=0.5, y=-0.1)
            plt.legend(handles=plt4.legend_elements()[0],
                       loc='best',
                       labels=['order', 'disorder'])  # handles=plot.legend_elements()[0],labels=['order', 'disorder']
            # if epoch>17:
            # plt.show()
            plt.savefig(os.path.join(final_output_dir, 'test_Epoch[{0}]_{1}.png'.format(epoch, i)), dpi=720)
            plt.close("all")


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

    loss_od_avg, loss_mol_avg ,loss_sum_avg= map(
        _meter_reduce if distributed else lambda x: x.avg,
        [losses_od, losses_mol, losses_sum]
    )

    if comm.is_main_process():
        msg = '=> TEST:\t' \
            'Loss_od {loss_od_avg:.4f}\t' \
            'Loss_mol {loss_mol_avg:.4f}\t'\
            'Loss_sum {loss_sum_avg:.4f}\t'.format(
                loss_od_avg=loss_od_avg, loss_mol_avg=loss_mol_avg, loss_sum_avg=loss_sum_avg
            )
        logging.info(msg)

    if writer_dict and comm.is_main_process():
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_od_acc', class_od_acc, global_steps)
        writer.add_scalar('valid_mol_acc', class_mol_acc, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    logging.info('=> switch to train mode')
    model.train()

    return instance_od_acc, instance_mol_acc

@torch.no_grad()
def test_multi_with_reg(final_output_dir, epoch, begin_epoch, end_epoch, config_od, config_mol, val_loader, model, criterion_cls, criterion_reg, output_dir, tb_log_dir,
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
        #pred_choice_mol = pred_mol.data.max(1)[1]

        loss_od = criterion_cls(pred_od, target_od.long())
        print(target_od)
        loss_mol = criterion_cls(pred_mol, target_mol.long())
        loss_temp = criterion_reg(pred_temp, target_temp.long())#.sum()
        print(target_temp)
        print(loss_temp)

        loss_sum = loss_od + loss_temp
        #loss_sum = loss_od + loss_mol + loss_temp

        losses_od.update(loss_od.item(), points.size(0))
        losses_mol.update(loss_mol.item(), points.size(0))
        losses_temp.update(loss_temp.item(), points.size(0))
        losses_sum.update(loss_sum.item(), points.size(0))

        if epoch == end_epoch+1:
            output_od = output_feature_l1_od.cpu().detach().numpy()  # .reshape(32, 1125*200)#.reshape(32, 3*200)
            output_mol = output_feature_l1_mol.cpu().detach().numpy()  # .reshape(32, 1125*200)#.reshape(32, 3*200)
            distortion_output = output_feature_l2.cpu().detach().numpy().reshape(256, 64 * 500)  # .reshape(32, 3*200)
            tsne2d = TSNE(n_components=2, init='pca', random_state=0)
            tsne_result_od = tsne2d.fit_transform(output_od)
            tsne_result_mol = tsne2d.fit_transform(output_mol)
            tsne_result_d = tsne2d.fit_transform(distortion_output)
            plt.figure(figsize=(26, 6))
            plot = plt.subplot(141)
            plt1 = plot.scatter(tsne_result_mol[:, 0], tsne_result_mol[:, 1],
                                c=target_mol.cpu().detach().numpy(), cmap='rainbow')  # c=pred_choice.cpu().detach().numpy()
            plt.title('mlp_mol_test', x=0.5, y=-0.1)
            plt.legend(handles=plt1.legend_elements()[0],
                       loc='best',
                       labels=['1188514',
                               '1286153',
                               '1142500',
                               '248562',
                               '1033994',
                               '1450800',
                               '1249410',
                               '1432562',
                               '2150172',
                               '1100214',
                               '1208853',
                               '1934968'])  # handles=plot.legend_elements()[0],labels=['1188514', '1286153']
            plot = plt.subplot(142)
            plt2 = plot.scatter(tsne_result_od[:, 0], tsne_result_od[:, 1],
                                c=target_od.cpu().detach().numpy())  # c=pred_choice.cpu().detach().numpy()
            plt.title('mlp_od_test', x=0.5, y=-0.1)
            plt.legend(handles=plt2.legend_elements()[0],
                       loc='best',
                       labels=['order', 'disorder'])  # handles=plot.legend_elements()[0],labels=['order', 'disorder']
            plot = plt.subplot(143)
            plt3 = plot.scatter(tsne_result_d[:, 0], tsne_result_d[:, 1],
                                c=target_mol.cpu().detach().numpy(), cmap='rainbow')  # c=pred_choice.cpu().detach().numpy()
            plt.title('conv_mol_test', x=0.5, y=-0.1)
            plt.legend(handles=plt3.legend_elements()[0], loc='best',
                       labels=['1188514',
                               '1286153',
                               '1142500',
                               '248562',
                               '1033994',
                               '1450800',
                               '1249410',
                               '1432562',
                               '2150172',
                               '1100214',
                               '1208853',
                               '1934968'])  # handles=plot.legend_elements()[0],labels=['1188514', '1286153']
            plot = plt.subplot(144)
            plt4 = plot.scatter(tsne_result_d[:, 0], tsne_result_d[:, 1],
                                c=target_od.cpu().detach().numpy())  # c=pred_choice.cpu().detach().numpy()
            plt.title('mlp_od_test', x=0.5, y=-0.1)
            plt.legend(handles=plt4.legend_elements()[0],
                       loc='best',
                       labels=['order', 'disorder'])  # handles=plot.legend_elements()[0],labels=['order', 'disorder']
            # if epoch>17:
            # plt.show()
            plt.savefig(os.path.join(final_output_dir, 'test_Epoch[{0}]_{1}.png'.format(epoch, i)), dpi=720)
            plt.close("all")


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        for cat in np.unique(target_od.cpu()):
            classacc_od = pred_choice_od[target_od == cat].eq(target_od[target_od == cat].long().data).cpu().sum()
            class_od_acc[cat, 0] += classacc_od.item() / float(points[target_od == cat].size()[0])
            class_od_acc[cat, 1] += 1

        #for cat in np.unique(target_mol.cpu()):
        #    classacc_mol = pred_choice_mol[target_mol == cat].eq(target_mol[target_mol == cat].long().data).cpu().sum()
        #    class_mol_acc[cat, 0] += classacc_mol.item() / float(points[target_mol == cat].size()[0])
        #    class_mol_acc[cat, 1] += 1

        correct_od = pred_choice_od.eq(target_od.long().data).cpu().sum()
        mean_correct_od.append(correct_od.item() / float(points.size()[0]))

        #correct_mol = pred_choice_mol.eq(target_mol.long().data).cpu().sum()
        #mean_correct_mol.append(correct_mol.item() / float(points.size()[0]))

    class_od_acc[:, 2] = class_od_acc[:, 0] / class_od_acc[:, 1]
    class_od_acc = np.mean(class_od_acc[:, 2])
    print('class_od_acc', class_od_acc)
    instance_od_acc = np.mean(mean_correct_od)
    logging.info(f'instance_od_acc {instance_od_acc:.3f}\t'.format(instance_od_acc=instance_od_acc))
    print('instance_od_acc', instance_od_acc)

    #class_mol_acc[:, 2] = class_mol_acc[:, 0] / class_mol_acc[:, 1]
    #class_mol_acc = np.mean(class_mol_acc[:, 2])
    #print('class_mol_acc', class_mol_acc)
    #instance_mol_acc = np.mean(mean_correct_mol)
    #logging.info(f'instance_mol_acc {instance_mol_acc:.3f}\t'.format(instance_mol_acc=instance_mol_acc))
    #print('instance_mol_acc', instance_mol_acc)

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

    return instance_od_acc, loss_temp_avg

@torch.no_grad()
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
        loss_temp = criterion_reg(pred_temp, target_temp.to(torch.float32))#.sum()
        # print(target_temp)
        print(pred_temp)
        print(loss_temp)

        #loss_sum = loss_od + loss_temp
        loss_sum = loss_od + loss_mol + loss_temp

        losses_od.update(loss_od.item(), points.size(0))
        losses_mol.update(loss_mol.item(), points.size(0))
        losses_temp.update(loss_temp.item(), points.size(0))
        losses_sum.update(loss_sum.item(), points.size(0))

        if epoch == end_epoch+1:
            output_od = output_feature_l1_od.cpu().detach().numpy()  # .reshape(32, 1125*200)#.reshape(32, 3*200)
            output_mol = output_feature_l1_mol.cpu().detach().numpy()  # .reshape(32, 1125*200)#.reshape(32, 3*200)
            distortion_output = output_feature_l2.cpu().detach().numpy().reshape(256, 64 * 500)  # .reshape(32, 3*200)
            tsne2d = TSNE(n_components=2, init='pca', random_state=0)
            tsne_result_od = tsne2d.fit_transform(output_od)
            tsne_result_mol = tsne2d.fit_transform(output_mol)
            tsne_result_d = tsne2d.fit_transform(distortion_output)
            plt.figure(figsize=(26, 6))
            plot = plt.subplot(141)
            plt1 = plot.scatter(tsne_result_mol[:, 0], tsne_result_mol[:, 1],
                                c=target_mol.cpu().detach().numpy(), cmap='rainbow')  # c=pred_choice.cpu().detach().numpy()
            plt.title('mlp_mol_test', x=0.5, y=-0.1)
            plt.legend(handles=plt1.legend_elements()[0],
                       loc='best',
                       labels=['1188514',
                               '1286153',
                               '1142500',
                               '248562',
                               '1033994',
                               '1450800',
                               '1249410',
                               '1432562',
                               '2150172',
                               '1100214',
                               '1208853',
                               '1934968'])  # handles=plot.legend_elements()[0],labels=['1188514', '1286153']
            plot = plt.subplot(142)
            plt2 = plot.scatter(tsne_result_od[:, 0], tsne_result_od[:, 1],
                                c=target_od.cpu().detach().numpy())  # c=pred_choice.cpu().detach().numpy()
            plt.title('mlp_od_test', x=0.5, y=-0.1)
            plt.legend(handles=plt2.legend_elements()[0],
                       loc='best',
                       labels=['order', 'disorder'])  # handles=plot.legend_elements()[0],labels=['order', 'disorder']
            plot = plt.subplot(143)
            plt3 = plot.scatter(tsne_result_d[:, 0], tsne_result_d[:, 1],
                                c=target_mol.cpu().detach().numpy(), cmap='rainbow')  # c=pred_choice.cpu().detach().numpy()
            plt.title('conv_mol_test', x=0.5, y=-0.1)
            plt.legend(handles=plt3.legend_elements()[0], loc='best',
                       labels=['1188514',
                               '1286153',
                               '1142500',
                               '248562',
                               '1033994',
                               '1450800',
                               '1249410',
                               '1432562',
                               '2150172',
                               '1100214',
                               '1208853',
                               '1934968'])  # handles=plot.legend_elements()[0],labels=['1188514', '1286153']
            plot = plt.subplot(144)
            plt4 = plot.scatter(tsne_result_d[:, 0], tsne_result_d[:, 1],
                                c=target_od.cpu().detach().numpy())  # c=pred_choice.cpu().detach().numpy()
            plt.title('mlp_od_test', x=0.5, y=-0.1)
            plt.legend(handles=plt4.legend_elements()[0],
                       loc='best',
                       labels=['order', 'disorder'])  # handles=plot.legend_elements()[0],labels=['order', 'disorder']
            # if epoch>17:
            # plt.show()
            plt.savefig(os.path.join(final_output_dir, 'test_Epoch[{0}]_{1}.png'.format(epoch, i)), dpi=720)
            plt.close("all")


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

@torch.no_grad()
def test_multi_with_mol_reg_dt(final_output_dir, epoch, begin_epoch, end_epoch, config_od, config_mol, val_loader, model, criterion_cls, criterion_reg, output_dir, tb_log_dir,
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

        pred_od, pred_mol, pred_dt, Tsys, output_feature_l2, output_feature_l1_od, output_feature_l1_mol = model(points)
        pred_choice_od = pred_od.data.max(1)[1]
        pred_choice_mol = pred_mol.data.max(1)[1]
        real_dt = Tsys.squeeze(-1) - target_temp

        loss_od = criterion_cls(pred_od, target_od.long())
        print(target_od)
        loss_mol = criterion_cls(pred_mol, target_mol.long())
        loss_temp = criterion_reg(pred_dt, real_dt.long())#.sum()
        print(target_temp)
        print(real_dt)
        print(loss_temp)

        #loss_sum = loss_od + loss_temp
        loss_sum = loss_od + loss_mol + loss_temp

        losses_od.update(loss_od.item(), points.size(0))
        losses_mol.update(loss_mol.item(), points.size(0))
        losses_temp.update(loss_temp.item(), points.size(0))
        losses_sum.update(loss_sum.item(), points.size(0))

        if epoch == end_epoch+1:
            output_od = output_feature_l1_od.cpu().detach().numpy()  # .reshape(32, 1125*200)#.reshape(32, 3*200)
            output_mol = output_feature_l1_mol.cpu().detach().numpy()  # .reshape(32, 1125*200)#.reshape(32, 3*200)
            distortion_output = output_feature_l2.cpu().detach().numpy().reshape(256, 64 * 500)  # .reshape(32, 3*200)
            tsne2d = TSNE(n_components=2, init='pca', random_state=0)
            tsne_result_od = tsne2d.fit_transform(output_od)
            tsne_result_mol = tsne2d.fit_transform(output_mol)
            tsne_result_d = tsne2d.fit_transform(distortion_output)
            plt.figure(figsize=(26, 6))
            plot = plt.subplot(141)
            plt1 = plot.scatter(tsne_result_mol[:, 0], tsne_result_mol[:, 1],
                                c=target_mol.cpu().detach().numpy(), cmap='rainbow')  # c=pred_choice.cpu().detach().numpy()
            plt.title('mlp_mol_test', x=0.5, y=-0.1)
            plt.legend(handles=plt1.legend_elements()[0],
                       loc='best',
                       labels=['1188514',
                               '1286153',
                               '1142500',
                               '248562',
                               '1033994',
                               '1450800',
                               '1249410',
                               '1432562',
                               '2150172',
                               '1100214',
                               '1208853',
                               '1934968'])  # handles=plot.legend_elements()[0],labels=['1188514', '1286153']
            plot = plt.subplot(142)
            plt2 = plot.scatter(tsne_result_od[:, 0], tsne_result_od[:, 1],
                                c=target_od.cpu().detach().numpy())  # c=pred_choice.cpu().detach().numpy()
            plt.title('mlp_od_test', x=0.5, y=-0.1)
            plt.legend(handles=plt2.legend_elements()[0],
                       loc='best',
                       labels=['order', 'disorder'])  # handles=plot.legend_elements()[0],labels=['order', 'disorder']
            plot = plt.subplot(143)
            plt3 = plot.scatter(tsne_result_d[:, 0], tsne_result_d[:, 1],
                                c=target_mol.cpu().detach().numpy(), cmap='rainbow')  # c=pred_choice.cpu().detach().numpy()
            plt.title('conv_mol_test', x=0.5, y=-0.1)
            plt.legend(handles=plt3.legend_elements()[0], loc='best',
                       labels=['1188514',
                               '1286153',
                               '1142500',
                               '248562',
                               '1033994',
                               '1450800',
                               '1249410',
                               '1432562',
                               '2150172',
                               '1100214',
                               '1208853',
                               '1934968'])  # handles=plot.legend_elements()[0],labels=['1188514', '1286153']
            plot = plt.subplot(144)
            plt4 = plot.scatter(tsne_result_d[:, 0], tsne_result_d[:, 1],
                                c=target_od.cpu().detach().numpy())  # c=pred_choice.cpu().detach().numpy()
            plt.title('mlp_od_test', x=0.5, y=-0.1)
            plt.legend(handles=plt4.legend_elements()[0],
                       loc='best',
                       labels=['order', 'disorder'])  # handles=plot.legend_elements()[0],labels=['order', 'disorder']
            # if epoch>17:
            # plt.show()
            plt.savefig(os.path.join(final_output_dir, 'test_Epoch[{0}]_{1}.png'.format(epoch, i)), dpi=720)
            plt.close("all")


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

@torch.no_grad()
def test_multi_with_reg_test(final_output_dir, epoch, begin_epoch, end_epoch, config_od, config_mol, val_loader, model, criterion_cls, criterion_reg, output_dir, tb_log_dir,
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
    sysT = []
    predT = []
    targetT = []

    end = time.time()
    for i, (points, target_mol, target_od, target_temp) in enumerate(val_loader):

        points = points.transpose(2, 1)
        # compute output
        points = points.cuda(non_blocking=True)
        target_mol = target_mol.cuda(non_blocking=True)
        target_od = target_od.cuda(non_blocking=True)
        target_temp = target_temp.cuda(non_blocking=True)

        pred_od, pred_mol, pred_temp, Tsys, output_feature_l2, output_feature_l1_od, output_feature_l1_mol, correlation, c, sa4, resl, resh = model(points)
        pred_choice_od = pred_od.data.max(1)[1]
        # print(pred_temp)

        # if i == 0:
        #     # tsne_od = output_feature_l1_od.cpu().detach().numpy()
        #     # tsne_mol = output_feature_l1_mol.cpu().detach().numpy()
        #     tsne_tod = target_od.cpu().detach().numpy()
        #     tsne_tmol = target_mol.cpu().detach().numpy()
        #     # tsne_d = output_feature_l2.cpu().detach().numpy().reshape(100, 128 * 500)
        #     # tsne_dsa4 = sa4.cpu().detach().numpy()
        #     tsne_res = resh.cpu().detach().numpy()
        #     # tsne_c = correlation.cpu().detach().numpy()
        #     # tsne_dc = c.cpu().detach().numpy()
        #     # tsne_dt = target_temp.cpu().detach().numpy() - Tsys.squeeze(-1).cpu().detach().numpy()
        #     tsne_dt = Tsys.squeeze(-1).cpu().detach().numpy() - target_temp.cpu().detach().numpy()
        #     # print(target_temp.cpu().detach().numpy().shape, Tsys.cpu().detach().numpy().shape, tsne_dt.shape)
        # else:
        #     # tsne_od = np.vstack((tsne_od, output_feature_l1_od.cpu().detach().numpy()))
        #     # tsne_mol = np.vstack((tsne_mol, output_feature_l1_mol.cpu().detach().numpy()))
        #     # tsne_d = np.vstack((tsne_d, output_feature_l2.cpu().detach().numpy().reshape(100, 128 * 500)))
        #     tsne_res = np.vstack((tsne_res, resh.cpu().detach().numpy()))
        #     # tsne_dsa4 = np.vstack((tsne_dsa4, sa4.cpu().detach().numpy()))
        #     tsne_tod = np.hstack((tsne_tod, target_od.cpu().detach().numpy()))
        #     tsne_tmol = np.hstack((tsne_tmol, target_mol.cpu().detach().numpy()))
        #     # tsne_c = np.vstack((tsne_c, correlation.cpu().detach().numpy()))
        #     # tsne_dc = np.vstack((tsne_dc, c.cpu().detach().numpy()))
        #     # tsne_dt = np.hstack((tsne_dt, target_temp.cpu().detach().numpy() - Tsys.squeeze(-1).cpu().detach().numpy()))
        #     tsne_dt = np.hstack((tsne_dt, Tsys.squeeze(-1).cpu().detach().numpy() - target_temp.cpu().detach().numpy()))
        #     # print(tsne_tod.shape, tsne_d.shape, tsne_c.shape, tsne_dt.shape)

        loss_od = criterion_cls(pred_od, target_od.long())
        # print(target_od)
        loss_mol = criterion_cls(pred_mol, target_mol.long())
        loss_temp = criterion_reg(pred_temp, target_temp.long())#.sum()
        print(loss_temp)
        # print(target_temp)
        # print(loss_temp)

        loss_sum = loss_od + loss_temp

        losses_od.update(loss_od.item(), points.size(0))
        losses_mol.update(loss_mol.item(), points.size(0))
        losses_temp.update(loss_temp.item(), points.size(0))
        losses_sum.update(loss_sum.item(), points.size(0))

        # sysT = np.append(sysT, Tsys.squeeze(-1).cpu().detach().numpy())
        # targetT = np.append(targetT, target_temp.cpu().detach().numpy())
        # predT = np.append(predT, pred_temp.cpu().detach().numpy())

        sysT = np.append(sysT, Tsys.squeeze(-1).cpu().detach().numpy().mean())
        targetT = np.append(targetT, target_temp.cpu().detach().numpy().mean())
        predT = np.append(predT, pred_temp.cpu().detach().numpy().mean())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        for cat in np.unique(target_od.cpu()):
            classacc_od = pred_choice_od[target_od == cat].eq(target_od[target_od == cat].long().data).cpu().sum()
            class_od_acc[cat, 0] += classacc_od.item() / float(points[target_od == cat].size()[0])
            class_od_acc[cat, 1] += 1

        correct_od = pred_choice_od.eq(target_od.long().data).cpu().sum()
        mean_correct_od.append(correct_od.item() / float(points.size()[0]))

    class_od_acc[:, 2] = class_od_acc[:, 0] / class_od_acc[:, 1]
    class_od_acc = np.mean(class_od_acc[:, 2])
    print('class_od_acc', class_od_acc)
    instance_od_acc = np.mean(mean_correct_od)
    logging.info(f'instance_od_acc {instance_od_acc:.3f}\t'.format(instance_od_acc=instance_od_acc))
    print('instance_od_acc', instance_od_acc)

    logging.info('=> synchronize...')
    comm.synchronize()

    # output_od = tsne_od  # .reshape(32, 1125*200)#.reshape(32, 3*200)
    # output_mol = tsne_mol
    # distortion_output = tsne_dsa4  # .reshape(32, 3*200)
    # res_output = tsne_res
    # correlation_output = tsne_c
    # dc_output = tsne_dc
    # tsne2d = TSNE(n_components=2, init='pca', random_state=0)
    # tsne_result_od = tsne2d.fit_transform(output_od)
    # tsne_result_mol = tsne2d.fit_transform(output_mol)
    # tsne_result_d = tsne2d.fit_transform(distortion_output)
    # tsne_result_c = tsne2d.fit_transform(correlation_output)
    # tsne_result_dc = tsne2d.fit_transform(dc_output)
    # ldt = np.zeros(len(tsne_dt))
    # # for item in range(len(tsne_dt)):
    # #     if tsne_dt[item] < 0 and tsne_dt[item] >= -10:
    # #         ldt[item] = -1
    # #     elif tsne_dt[item] < -10 and tsne_dt[item] >= -20:
    # #         ldt[item] = -2
    # #     elif tsne_dt[item] < -20 and tsne_dt[item] >= -30:
    # #         ldt[item] = -3
    # #     elif tsne_dt[item] < -30 and tsne_dt[item] >= -40:
    # #         ldt[item] = -4
    # #     elif tsne_dt[item] < -40 and tsne_dt[item] >= -50:
    # #         ldt[item] = -5
    # #     elif tsne_dt[item] < -50 and tsne_dt[item] >= -60:
    # #         ldt[item] = -6
    # #     elif tsne_dt[item] < -60 and tsne_dt[item] >= -70:
    # #         ldt[item] = -7
    # #     elif tsne_dt[item] < -70 and tsne_dt[item] >= -80:
    # #         ldt[item] = -8
    # #     elif tsne_dt[item] < -80 and tsne_dt[item] >= -90:
    # #         ldt[item] = -9
    # #     elif tsne_dt[item] > 0 and tsne_dt[item] <= 10:
    # #         ldt[item] = 1
    # #     elif tsne_dt[item] > 10 and tsne_dt[item] <= 20:
    # #         ldt[item] = 2
    # #     elif tsne_dt[item] > 20 and tsne_dt[item] <= 30:
    # #         ldt[item] = 3
    # #     elif tsne_dt[item] > 30 and tsne_dt[item] <= 40:
    # #         ldt[item] = 4
    # #     elif tsne_dt[item] > 40 and tsne_dt[item] <= 50:
    # #         ldt[item] = 5
    # #     elif tsne_dt[item] > 50 and tsne_dt[item] <= 60:
    # #         ldt[item] = 6
    # #     elif tsne_dt[item] > 60 and tsne_dt[item] <= 70:
    # #         ldt[item] = 7
    # #     elif tsne_dt[item] > 70 and tsne_dt[item] <= 80:
    # #         ldt[item] = 8
    # #     elif tsne_dt[item] > 80 and tsne_dt[item] <= 90:
    # #         ldt[item] = 9
    # pca = PCA(n_components=0.95)
    # pca_c = StandardScaler().fit_transform(correlation_output)
    # pca_dc = StandardScaler().fit_transform(dc_output)
    # pca_distortion = StandardScaler().fit_transform(distortion_output)
    # pca_res = StandardScaler().fit_transform(res_output)
    # pca_mol = StandardScaler().fit_transform(output_mol)
    # pca_result_c = pca.fit_transform(pca_c)
    # pca_result_dc = pca.fit_transform(pca_dc)
    # pca_result_d = pca.fit_transform(pca_distortion)
    # pca_result_res = pca.fit_transform(pca_res)
    # pca_result_mol = pca.fit_transform(pca_mol)
    #
    #

    # plt.figure(figsize=(2.4, 2))
    # ax = plt.scatter(pca_result_res[:, 0], pca_result_res[:, 1], s=0.3, lw=0, c=tsne_dt, cmap='plasma', vmin=0, vmax=60)  # c=pred_choice.cpu().detach().numpy()
    # plt.xlabel('PCA component 1', fontsize=5, va='center')
    # plt.ylabel('PCA component 2', fontsize=5, va='center')
    # plt.title('Extended feature PCA', fontsize=5)
    # plt.tick_params(axis='both', length=2, pad=2, labelsize=5)
    # plt.xlim([-25, 30])
    # plt.xticks([-20, -10, 0, 10, 20, 30])
    # plt.ylim([-5, 15])
    # plt.yticks([-5, 0, 5, 10, 15])
    # clb = plt.colorbar(ax, ticks=[10, 20, 30, 40, 50])
    # # plt.legend(handles=ax.legend_elements()[0], loc=2, labels=['order', 'disorder'], fontsize=5, markerscale=0.3)
    # clb.set_label('TBMP (K)', fontsize=5, rotation=270, labelpad=8.5)
    # clb.ax.tick_params(labelsize=5)
    # # plt.axis('equal')
    # plt.tight_layout()
    # plt.savefig(os.path.join(final_output_dir, 'pca_total_temp_liquid2.png'), dpi=300)

    # plt.figure(figsize=(2.4, 2))
    # ax = plt.scatter(pca_result_res[:, 0], pca_result_res[:, 1], s=0.3, lw=0, c=tsne_dt, cmap='plasma', vmin=-60, vmax=0)  # c=pred_choice.cpu().detach().numpy()
    # plt.xlabel('PCA component 1', fontsize=5, va='center')
    # plt.ylabel('PCA component 2', fontsize=5, va='center')
    # plt.title('Extended feature PCA', fontsize=5)
    # plt.tick_params(axis='both', length=2, pad=2, labelsize=5)
    # plt.xlim([-25, 28])
    # plt.xticks([-20, -10, 0, 10, 20])
    # plt.ylim([-5, 15])
    # plt.yticks([-5, 0, 5, 10, 15])
    # clb = plt.colorbar(ax, ticks=[-50, -40, -30, -20, -10])
    # # plt.legend(handles=ax.legend_elements()[0], loc=2, labels=['order', 'disorder'], fontsize=5, markerscale=0.3)
    # clb.set_label('TBMP (K)', fontsize=5, rotation=270, labelpad=8.5)
    # clb.ax.tick_params(labelsize=5)
    # # plt.axis('equal')
    # plt.tight_layout()
    # plt.savefig(os.path.join(final_output_dir, 'pca_total_temp_solid2.png'), dpi=300)

    # plt.figure(figsize=(2, 2))
    # ax = plt.scatter(pca_result_c[:, 0], pca_result_c[:, 1], s=0.3, lw=0, c=tsne_dt, cmap='plasma', vmin=-60, vmax=60)  # c=pred_choice.cpu().detach().numpy()
    # plt.xlabel('PCA component 1', fontsize=5, va='center')
    # plt.ylabel('PCA component 2', fontsize=5, va='center')
    # plt.title('Correlation PCA', fontsize=5)
    # plt.tick_params(axis='both', length=2, pad=2, labelsize=5)
    # # plt.xlim([-15, 15])
    # # plt.xticks([-15, -10, -5, 0, 5, 10, 15])
    # # plt.ylim([-10, 15])
    # # plt.yticks([-10, -5, 0, 5, 10, 15])
    # clb = plt.colorbar(ax, ticks=[-50, -40, -30, -20, -10, 10, 20, 30, 40, 50])
    # # plt.legend(handles=ax.legend_elements()[0], loc=2, labels=['order', 'disorder'], fontsize=5, markerscale=0.3)
    # clb.set_label('TBMP (K)', fontsize=5, rotation=270, labelpad=8.5)
    # clb.ax.tick_params(labelsize=5)
    # # plt.axis('square')
    # plt.tight_layout()
    # plt.savefig(os.path.join(final_output_dir, 'pca_correlation_temp2.png'), dpi=300)

    # plt.figure(figsize=(2, 2))
    # plt.scatter(tsne_result_d[:, 0], tsne_result_d[:, 1], c=tsne_tmol, cmap='rainbow', s=0.3, lw=0)  # c=pred_choice.cpu().detach().numpy()
    # plt.xlabel('t-SNE component 1', fontsize=5, va='center')
    # plt.ylabel('t-SNE component 2', fontsize=5, va='center')
    # plt.title('Global feature TSNE', fontsize=5)
    # # plt.xticks([-20, -15, -10, -5, 0, 5, 10, 15, 20])
    # # plt.yticks([-20, -15, -10, -5, 0, 5, 10, 15, 20])
    # plt.tick_params(axis='both', length=2, pad=2, labelsize=5)
    # # plt.legend(handles=ax.legend_elements()[0], loc=2, labels=['order', 'disorder'], fontsize=5, markerscale=0.3)
    # # plt.axis('square')
    # plt.tight_layout()
    # plt.savefig(os.path.join(final_output_dir, 'tsne_global_feature_mol2.png'), dpi=300)
    #
    # plt.figure(figsize=(2, 2))
    # ax = plt.scatter(tsne_result_d[:, 0], tsne_result_d[:, 1], c=tsne_tod, s=0.3, lw=0)  # c=pred_choice.cpu().detach().numpy()
    # plt.xlabel('t-SNE component 1', fontsize=5, va='center')
    # plt.ylabel('t-SNE component 2', fontsize=5, va='center')
    # plt.title('Global feature TSNE', fontsize=5)
    # # plt.xticks([-20, -15, -10, -5, 0, 5, 10, 15, 20])
    # # plt.yticks([-20, -15, -10, -5, 0, 5, 10, 15, 20])
    # plt.tick_params(axis='both', length=2, pad=2, labelsize=5)
    # plt.legend(handles=ax.legend_elements()[0], loc=2, labels=['Crystal', 'Liquid'], fontsize=3, markerscale=0.3, handletextpad=0.1, handlelength=2)
    # # plt.axis('square')
    # plt.tight_layout()
    # plt.savefig(os.path.join(final_output_dir, 'tsne_global_feature_od2.png'), dpi=300)
    #
    # plt.figure(figsize=(2, 2))
    # plt.scatter(pca_result_d[:, 0], pca_result_d[:, 1], c=tsne_tmol, cmap='rainbow', s=0.3, lw=0)  # c=pred_choice.cpu().detach().numpy()
    # plt.xlabel('PCA component 1', fontsize=5, va='center')
    # plt.ylabel('PCA component 2', fontsize=5, va='center')
    # plt.title('Global feature PCA', fontsize=5)
    # plt.xticks([-20, -15, -10, -5, 0, 5, 10, 15, 20])
    # plt.yticks([-20, -15, -10, -5, 0, 5, 10, 15, 20])
    # plt.tick_params(axis='both', length=2, pad=2, labelsize=5)
    # # plt.axis('square')
    # plt.tight_layout()
    # plt.savefig(os.path.join(final_output_dir, 'pca_global_feature_mol2.png'), dpi=300)
    #
    # plt.figure(figsize=(2, 2))
    # ax = plt.scatter(pca_result_d[:, 0], pca_result_d[:, 1], c=tsne_tod, s=0.3, lw=0)  # c=pred_choice.cpu().detach().numpy()
    # plt.xlabel('PCA component 1', fontsize=5, va='center')
    # plt.ylabel('PCA component 2', fontsize=5, va='center')
    # plt.title('Global feature PCA', fontsize=5)
    # plt.xticks([-20, -15, -10, -5, 0, 5, 10, 15, 20])
    # plt.yticks([-20, -15, -10, -5, 0, 5, 10, 15, 20])
    # plt.tick_params(axis='both', length=2, pad=2, labelsize=5)
    # plt.legend(handles=ax.legend_elements()[0], loc=2, labels=['order', 'disorder'], fontsize=5, markerscale=0.3)
    # # plt.axis('square')
    # plt.tight_layout()
    # plt.savefig(os.path.join(final_output_dir, 'pca_global_feature_od2.png'), dpi=300)

    # plt.figure(figsize=(54, 6))
    # # plt.rcParams['font.sans-serif'] = ['Times New Roman']
    # plot = plt.subplot(181)
    # plot.set_aspect('equal', adjustable='box')
    # plt1 = plot.scatter(tsne_result_mol[:, 0], tsne_result_mol[:, 1],
    #                     c=tsne_tmol, cmap='rainbow', s=20)  # c=pred_choice.cpu().detach().numpy()
    # plt.title('mlp_mol_test', x=0.5, y=-0.1)
    # # plt.legend(handles=plt3.legend_elements()[0],
    # #            loc='best')  # handles=plot.legend_elements()[0],labels=['1188514', '1286153']
    # plot = plt.subplot(182)
    # plot.set_aspect('equal', adjustable='box')
    # plt2 = plot.scatter(tsne_result_od[:, 0], tsne_result_od[:, 1],
    #                     c=tsne_tod, s=20)  # c=pred_choice.cpu().detach().numpy()
    # plt.title('mlp_od_test', x=0.5, y=-0.1)
    # plt.legend(handles=plt2.legend_elements()[0],
    #            loc='best',
    #            labels=['order', 'disorder'])  # handles=plot.legend_elements()[0],labels=['order', 'disorder']
    # plot = plt.subplot(183)
    # plot.set_aspect('equal', adjustable='box')
    # plt3 = plot.scatter(tsne_result_d[:, 0], tsne_result_d[:, 1],
    #                     c=tsne_tmol, cmap='rainbow', s=20)  # c=pred_choice.cpu().detach().numpy()
    # plt.title('conv_mol_test', x=0.5, y=-0.1)
    # # plt.legend(handles=plt3.legend_elements()[0],
    # #            loc='best')  # handles=plot.legend_elements()[0],labels=['1188514', '1286153']
    # plot = plt.subplot(184)
    # plot.set_aspect('equal', adjustable='box')
    # plt4 = plot.scatter(tsne_result_d[:, 0], tsne_result_d[:, 1],
    #                     c=tsne_tod, s=20)  # c=pred_choice.cpu().detach().numpy()
    # plt.title('conv_od_test', x=0.5, y=-0.1)
    # plt.legend(handles=plt4.legend_elements()[0],
    #            loc='best',
    #            labels=['order', 'disorder'])  # handles=plot.legend_elements()[0],labels=['order', 'disorder']
    # plot = plt.subplot(185)
    # plot.set_aspect('equal', adjustable='box')
    # plt5 = plot.scatter(tsne_result_c[:, 0], tsne_result_c[:, 1],
    #                     c=tsne_dt, cmap='plasma', vmin=-60, vmax=60, s=20)
    # plt.colorbar(plt5, ticks=[-50,-40,-30,-20,-10,10,20,30,40,50], shrink=0.8)# c=pred_choice.cpu().detach().numpy()
    # # plt.legend(handles=plt5.legend_elements()[0],
    # #            loc='best')
    # plt.title('correlation', x=0.5, y=-0.2)
    # plot = plt.subplot(186)
    # plot.set_aspect('equal', adjustable='box')
    # plt6 = plot.scatter(tsne_result_dc[:, 0], tsne_result_dc[:, 1],
    #                     c=tsne_dt, cmap='plasma', vmin=-60, vmax=60, s=20)  # c=pred_choice.cpu().detach().numpy()
    # plt.colorbar(plt6, ticks=[-50,-40,-30,-20,-10,10,20,30,40,50], shrink=0.8)
    # # plt.legend(handles=plt6.legend_elements()[0],
    # #            loc='best')
    # plt.title('correlation', x=0.5, y=-0.2)
    # plot = plt.subplot(187)
    # plot.set_aspect('equal', adjustable='box')
    # plt7 = plot.scatter(pca_result_c[:, 0], pca_result_c[:, 1],
    #                     c=tsne_dt, cmap='plasma', vmin=-60, vmax=60, s=5)
    # plt.colorbar(plt7, ticks=[-50,-40,-30,-20,-10,10,20,30,40,50])# c=pred_choice.cpu().detach().numpy()
    # # plt.legend(handles=plt5.legend_elements()[0],
    # #            loc='best')
    # plt.title('correlation', x=0.5, y=-0.1)
    # plot = plt.subplot(188)
    # plot.set_aspect('equal', adjustable='box')
    # plt8 = plot.scatter(pca_result_dc[:, 0], pca_result_dc[:, 1],
    #                     c=tsne_dt, cmap='plasma', vmin=-60, vmax=60, s=1)  # c=pred_choice.cpu().detach().numpy()
    # plt.colorbar(plt8, ticks=[-50,-40,-30,-20,-10,10,20,30,40,50], shrink=0.5)
    # # plt.legend(handles=plt6.legend_elements()[0],
    # #            loc='best')
    # plt.title('correlation', x=0.5, y=-0.2)
    # # plot = plt.subplot(254)
    # # plot.set_aspect('equal', adjustable='box')
    # # plt9 = plot.scatter(pca_result_mol[:, 0], pca_result_mol[:, 1],
    # #                     c=tsne_tmol, cmap='rainbow')  # c=pred_choice.cpu().detach().numpy()
    # # plt.title('mlp_mol_test', x=0.5, y=-0.1)
    # # plot = plt.subplot(255)
    # # plot.set_aspect('equal', adjustable='box')
    # # plt10 = plot.scatter(pca_result_d[:, 0], pca_result_d[:, 1],
    # #                     c=tsne_tmol, cmap='rainbow')  # c=pred_choice.cpu().detach().numpy()
    # # plt.title('conv_mol_test', x=0.5, y=-0.1)
    #
    #
    # # if epoch>17:
    # # plt.show()
    # plt.savefig(os.path.join(final_output_dir, 'test_Epoch[{0}]_{1}.png'), dpi=300)
    # plt.close("all")

    # plt.figure(figsize=(18, 6))
    # plot = plt.subplot(121)
    # plt1 = plot.scatter(pca_result_d[:, 0], pca_result_d[:, 1],
    #                     c=tsne_tmol, cmap='rainbow')  # c=pred_choice.cpu().detach().numpy()
    # plt.title('conv_mol_test', x=0.5, y=-0.1)
    # plot = plt.subplot(122)
    # plt2 = plot.scatter(pca_result_d[:, 0], pca_result_d[:, 1],
    #                     c=tsne_tod)  # c=pred_choice.cpu().detach().numpy()
    # plt.title('conv_od_test', x=0.5, y=-0.1)
    # plt.legend(handles=plt2.legend_elements()[0],
    #                       loc='best',
    #                       labels=['order', 'disorder'])  # handles=plot.legend_elements()[0],labels=['order', 'disorder']
    # plt.savefig(os.path.join(final_output_dir, 'pca.png'), dpi=300)
    # plt.close("all")

    loss_od_avg, loss_mol_avg, loss_temp_avg, loss_sum_avg= map(
        _meter_reduce if distributed else lambda x: x.avg,
        [losses_od, losses_mol, losses_temp, losses_sum]
    )

    loss_od_avg, loss_mol_avg, loss_temp_avg, loss_sum_avg= map(
        _meter_reduce if distributed else lambda x: x.avg,
        [losses_od, losses_mol, losses_temp, losses_sum]
    )

    sysT = np.array(sysT).reshape(-1)
    targetT = np.array(targetT).reshape(-1)
    predT = np.array(predT).reshape(-1)

    # deltaT_pred = sysT - predT
    # deltaT_real = sysT - targetT
    #
    # mae = mean_absolute_error(deltaT_real, deltaT_pred)
    # rmse = np.sqrt(mean_squared_error(deltaT_real, deltaT_pred))
    # r_2 = r2_score(deltaT_real, deltaT_pred)

    mae = mean_absolute_error(predT, targetT)
    rmse = np.sqrt(mean_squared_error(predT, targetT))
    r_2 = r2_score(predT, targetT)

    # x45 = np.linspace(-50.0, 50.0, 100)
    # y45=x45
    #
    # plt.figure(figsize=(2, 2))
    # plt.plot(x45, y45, c='r', linewidth=0.5)
    # plt.scatter(deltaT_real, deltaT_pred, s=1)  # c=pred_choice.cpu().detach().numpy()
    # plt.text(-53, 50, s=f"$R^2$={round(r_2, 2)}", fontsize=5)
    # plt.text(-53, 42, 'rmse={:.2f}'.format(rmse), fontsize=5)
    # plt.text(-53, 34, 'mae={:.2f}'.format(mae), fontsize=5)
    # # plt.axvline(-47, color='black', linestyle='--') #shu
    # # plt.axvline(-37, color='black', linestyle='--')
    # # plt.axvline(-27, color='black', linestyle='--')
    # # plt.axvline(-17, color='black', linestyle='--')
    # # plt.axvline(13, color='black', linestyle='--')
    # # plt.axvline(23, color='black', linestyle='--')
    # # plt.axvline(33, color='black', linestyle='--')
    # # plt.axvline(43, color='black', linestyle='--')
    # # plt.axhline(-47, color='black', linestyle='--') #hen
    # # plt.axhline(-37, color='black', linestyle='--')
    # # plt.axhline(-27, color='black', linestyle='--')
    # # plt.axhline(-17, color='black', linestyle='--')
    # # plt.axhline(13, color='black', linestyle='--') #hen
    # # plt.axhline(23, color='black', linestyle='--')
    # # plt.axhline(33, color='black', linestyle='--')
    # # plt.axhline(43, color='black', linestyle='--')
    # plt.xlim([-50.0, 50.0])
    # plt.xticks([-40,-20,0,20,40])
    # plt.ylim([-50.0, 50.0])
    # plt.yticks([-40,-20,0,20,40])
    # plt.tick_params(axis='both', length=2, pad=2, labelsize=5)
    # plt.axis('square')
    # plt.xlabel('True TBMP (K)', fontsize=5, va='center')
    # plt.ylabel('Predicted TBMP (K)', fontsize=5, va='center')
    # plt.tight_layout()
    #
    # plt.savefig(os.path.join(final_output_dir, 'test_Epoch_rmse_113810.png'), dpi=300)

    # x45 = np.linspace(0, 60.0, 100)
    # y45=x45
    #
    # plt.figure(figsize=(2, 2))
    # plt.scatter(deltaT_real, deltaT_pred, s=1)  # c=pred_choice.cpu().detach().numpy()
    # plt.text(-1, 61, s=f"$R^2$={round(r_2, 2)}", fontsize=5)
    # plt.text(-1, 56, 'rmse={:.2f}'.format(rmse), fontsize=5)
    # plt.text(-1, 51, 'mae={:.2f}'.format(mae), fontsize=5)
    # plt.plot(x45, y45, c='r', linewidth=0.5)
    # plt.xlim([0, 60.0])
    # plt.xticks([0,20,40,60])
    # plt.ylim([0, 60.0])
    # plt.yticks([0,20,40,60])
    # plt.tick_params(axis='both', length=2, pad=2, labelsize=5)
    # plt.axis('square')
    # plt.xlabel('True TBMP (K)', fontsize=5, va='center')
    # plt.ylabel('Predicted TBMP (K)', fontsize=5, va='center')
    # plt.tight_layout()
    #
    # plt.savefig(os.path.join(final_output_dir, 'valid.png'), dpi=300)
    # plt.close("all")

    x45 = np.linspace(280.0, 460.0, 180)
    y45=x45
    Ttest = targetT[0:70] #s76l70
    TPtest = predT[0:70]
    print(TPtest.shape)
    Tvalid = targetT[70:]
    TPvalid = predT[70:]
    print(TPvalid.shape)
    #
    # maev = mean_absolute_error(TPtest, Ttest)
    # rmsev = np.sqrt(mean_squared_error(TPtest, Ttest))
    # r_2v = r2_score(TPtest, Ttest)
    # print("validation", rmsev, maev, r_2v)
    #
    # maet = mean_absolute_error(TPvalid, Tvalid)
    # rmset = np.sqrt(mean_squared_error(TPvalid, Tvalid))
    # r_2t = r2_score(TPvalid, Tvalid)
    # print("test", rmset, maet, r_2t)

    plt.figure(figsize=(2, 2))
    plt.scatter(Ttest, TPtest , s=1, marker='s', label='Validation')  # c=pred_choice.cpu().detach().numpy()
    plt.scatter(Tvalid, TPvalid, c='orange', s=1, marker='^', label='Test')  # c=pred_choice.cpu().detach().numpy()
    plt.text(275, 458, s=f"$R^2$={round(r_2, 2)}", fontsize=5)
    plt.text(275, 445, 'RMSE={:.2f}'.format(rmse), fontsize=5)
    plt.text(275, 431, 'MAE={:.2f}'.format(mae), fontsize=5)
    plt.plot(x45, y45, c='r', linewidth=0.5)
    plt.xlim([280, 460])
    plt.xticks(np.arange(280,461,30))
    plt.ylim([280, 460])
    plt.yticks(np.arange(280,461,30))
    plt.tick_params(axis='both', length=2, pad=2, labelsize=5)
    plt.axis('square')
    plt.xlabel('True MP (K)', fontsize=5, va='center')
    plt.ylabel('Predicted MP (K)', fontsize=5, va='center')
    plt.legend(loc=4, fontsize=5)
    plt.tight_layout()

    plt.savefig(os.path.join(final_output_dir, 'test_rmse_liquid2.png'), dpi=300)
    plt.close("all")

    # x45 = np.linspace(-100.0, 100.0, 200)
    # y45=x45
    #
    # plt.figure(figsize=(6, 6))
    # plt.scatter(deltaT_real, deltaT_pred, c='r', s=1)  # c=pred_choice.cpu().detach().numpy()
    # plt.text( -65, 65, 'R2={:.2f}'.format(r_2))
    # plt.text( -65, 60, 'rmse={:.2f}'.format(rmse))
    # plt.text( -65, 55, 'mae={:.2f}'.format(mae))
    # plt.plot(x45,y45)
    # plt.axvline(-40, color='black', linestyle='--') #shu
    # plt.axvline(-30, color='black', linestyle='--')
    # plt.axvline(-20, color='black', linestyle='--')
    # plt.axvline(-10, color='black', linestyle='--')
    # plt.axvline(20, color='black', linestyle='--')
    # plt.axvline(30, color='black', linestyle='--')
    # plt.axvline(40, color='black', linestyle='--')
    # plt.axvline(50, color='black', linestyle='--')
    # plt.axhline(-40, color='black', linestyle='--') #hen
    # plt.axhline(-30, color='black', linestyle='--')
    # plt.axhline(-20, color='black', linestyle='--')
    # plt.axhline(-10, color='black', linestyle='--')
    # plt.axhline(20, color='black', linestyle='--') #hen
    # plt.axhline(30, color='black', linestyle='--')
    # plt.axhline(40, color='black', linestyle='--')
    # plt.axhline(50, color='black', linestyle='--')
    # plt.xlim([-70.0, 70.0])
    # plt.ylim([-70.0, 70.0])
    # plt.xlabel('deltaT_real')
    # plt.ylabel('deltaT_predict')
    #
    # plt.savefig(os.path.join(final_output_dir, 'test_Epoch_rmse_1306598.png'), dpi=720)
    # plt.close("all")

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

    return instance_od_acc, loss_temp_avg

@torch.no_grad()
def test_multi_with_reg_test_dt(final_output_dir, epoch, begin_epoch, end_epoch, config_od, config_mol, val_loader, model, criterion_cls, criterion_reg, output_dir, tb_log_dir,
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
    sysT = []
    preddT = []
    targetdT = []

    end = time.time()
    for i, (points, target_mol, target_od, target_temp) in enumerate(val_loader):

        points = points.transpose(2, 1)
        # compute output
        points = points.cuda(non_blocking=True)
        target_mol = target_mol.cuda(non_blocking=True)
        target_od = target_od.cuda(non_blocking=True)
        target_temp = target_temp.cuda(non_blocking=True)

        pred_od, pred_mol, pred_dt, Tsys, output_feature_l2, output_feature_l1_od, output_feature_l1_mol = model(points)
        pred_choice_od = pred_od.data.max(1)[1]
        real_dt = Tsys.squeeze(-1) - target_temp

        loss_od = criterion_cls(pred_od, target_od.long())
        print(target_od)
        loss_mol = criterion_cls(pred_mol, target_mol.long())
        loss_temp = criterion_reg(pred_dt, real_dt.long())#.sum()
        print(target_temp)
        print(loss_temp)

        loss_sum = loss_od + loss_temp

        losses_od.update(loss_od.item(), points.size(0))
        losses_mol.update(loss_mol.item(), points.size(0))
        losses_temp.update(loss_temp.item(), points.size(0))
        losses_sum.update(loss_sum.item(), points.size(0))

        sysT = np.append(sysT, Tsys.squeeze(-1).cpu().detach().numpy())
        targetdT = np.append(targetdT, real_dt.cpu().detach().numpy())
        preddT = np.append(preddT, pred_dt.cpu().detach().numpy())

        if epoch == -1:
            output_od = output_feature_l1_od.cpu().detach().numpy()  # .reshape(32, 1125*200)#.reshape(32, 3*200)
            output_mol = output_feature_l1_mol.cpu().detach().numpy()  # .reshape(32, 1125*200)#.reshape(32, 3*200)
            distortion_output = output_feature_l2.cpu().detach().numpy().reshape(256, 64 * 500)  # .reshape(32, 3*200)
            tsne2d = TSNE(n_components=2, init='pca', random_state=0)
            tsne_result_od = tsne2d.fit_transform(output_od)
            tsne_result_mol = tsne2d.fit_transform(output_mol)
            tsne_result_d = tsne2d.fit_transform(distortion_output)
            plt.figure(figsize=(26, 6))
            plot = plt.subplot(141)
            plt1 = plot.scatter(tsne_result_mol[:, 0], tsne_result_mol[:, 1],
                                c=target_mol.cpu().detach().numpy(), cmap='rainbow')  # c=pred_choice.cpu().detach().numpy()
            plt.title('mlp_mol_test', x=0.5, y=-0.1)
            plt.legend(handles=plt1.legend_elements()[0],
                       loc='best',
                       labels=['1286153',
                               '1033994',
                               '2150172'])  # handles=plot.legend_elements()[0],labels=['1188514', '1286153']
            plot = plt.subplot(142)
            plt2 = plot.scatter(tsne_result_od[:, 0], tsne_result_od[:, 1],
                                c=target_od.cpu().detach().numpy())  # c=pred_choice.cpu().detach().numpy()
            plt.title('mlp_od_test', x=0.5, y=-0.1)
            plt.legend(handles=plt2.legend_elements()[0],
                       loc='best',
                       labels=['order', 'disorder'])  # handles=plot.legend_elements()[0],labels=['order', 'disorder']
            plot = plt.subplot(143)
            plt3 = plot.scatter(tsne_result_d[:, 0], tsne_result_d[:, 1],
                                c=target_mol.cpu().detach().numpy(), cmap='rainbow')  # c=pred_choice.cpu().detach().numpy()
            plt.title('conv_mol_test', x=0.5, y=-0.1)
            plt.legend(handles=plt3.legend_elements()[0], loc='best',
                       labels=['1286153',
                               '1033994',
                               '2150172'])  # handles=plot.legend_elements()[0],labels=['1188514', '1286153']
            plot = plt.subplot(144)
            plt4 = plot.scatter(tsne_result_d[:, 0], tsne_result_d[:, 1],
                                c=target_od.cpu().detach().numpy())  # c=pred_choice.cpu().detach().numpy()
            plt.title('mlp_od_test', x=0.5, y=-0.1)
            plt.legend(handles=plt4.legend_elements()[0],
                       loc='best',
                       labels=['order', 'disorder'])  # handles=plot.legend_elements()[0],labels=['order', 'disorder']
            # if epoch>17:
            # plt.show()
            plt.savefig(os.path.join(final_output_dir, 'test_Epoch[{0}]_{1}.png'.format(epoch, i)), dpi=720)
            plt.close("all")


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        for cat in np.unique(target_od.cpu()):
            classacc_od = pred_choice_od[target_od == cat].eq(target_od[target_od == cat].long().data).cpu().sum()
            class_od_acc[cat, 0] += classacc_od.item() / float(points[target_od == cat].size()[0])
            class_od_acc[cat, 1] += 1

        correct_od = pred_choice_od.eq(target_od.long().data).cpu().sum()
        mean_correct_od.append(correct_od.item() / float(points.size()[0]))

    class_od_acc[:, 2] = class_od_acc[:, 0] / class_od_acc[:, 1]
    class_od_acc = np.mean(class_od_acc[:, 2])
    print('class_od_acc', class_od_acc)
    instance_od_acc = np.mean(mean_correct_od)
    logging.info(f'instance_od_acc {instance_od_acc:.3f}\t'.format(instance_od_acc=instance_od_acc))
    print('instance_od_acc', instance_od_acc)

    logging.info('=> synchronize...')
    comm.synchronize()

    loss_od_avg, loss_mol_avg, loss_temp_avg, loss_sum_avg= map(
        _meter_reduce if distributed else lambda x: x.avg,
        [losses_od, losses_mol, losses_temp, losses_sum]
    )

    sysT = np.array(sysT).reshape(-1)
    targetdT = np.array(targetdT).reshape(-1)
    preddT = np.array(preddT).reshape(-1)

    #deltaT_pred = predT - sysT
    #deltaT_real = targetT - sysT

    mae = mean_absolute_error(targetdT, preddT)
    rmse = np.sqrt(mean_squared_error(targetdT, preddT))
    r_2 = r2_score(targetdT, preddT)

    plt.figure(figsize=(6, 6))
    plt.scatter(targetdT, preddT, c='r', s=1)  # c=pred_choice.cpu().detach().numpy()
    plt.text(.5, 20, 'R2={:.2f}'.format(r_2))
    plt.text(.5, 18, 'rmse={:.2f}'.format(rmse))
    plt.text(.5, 16, 'mae={:.2f}'.format(mae))
    #plt.xlim([-80.0, 80.0])
    #plt.ylim([-80.0, 80.0])
    plt.xlabel('deltaT_real')
    plt.ylabel('deltaT_predict')

    plt.savefig(os.path.join(final_output_dir, 'test_Epoch_rmse_1286153[{0}]_{1}.png'.format(epoch, i)), dpi=720)
    plt.close("all")

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

    return instance_od_acc, loss_temp_avg

@torch.no_grad()
def test_multi_with_reg_simple(final_output_dir, epoch, begin_epoch, end_epoch, config_od, config_mol, val_loader, model, criterion_cls, criterion_reg, output_dir, tb_log_dir,
         writer_dict=None, distributed=False): #final_output_dir, epoch,
    batch_time = AverageMeter()
    mean_correct_od = []
    class_od_acc = np.zeros((config_od, 3))
    losses_od = AverageMeter()
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
        target_od = target_od.cuda(non_blocking=True)
        target_mol = target_mol.cuda(non_blocking=True)
        target_temp = target_temp.cuda(non_blocking=True)

        pred_od, pred_temp, Tsys, output_feature_l2, output_feature_l1_od = model(points)
        pred_choice_od = pred_od.data.max(1)[1]

        loss_od = criterion_cls(pred_od, target_od.long())
        # print(target_od)
        loss_temp = criterion_reg(pred_temp, target_temp.long())#.sum()
        # print(target_temp)
        print(loss_temp)

        loss_sum = loss_od + loss_temp

        losses_od.update(loss_od.item(), points.size(0))
        losses_temp.update(loss_temp.item(), points.size(0))
        losses_sum.update(loss_sum.item(), points.size(0))

        if epoch == end_epoch+1:
            output_od = output_feature_l1_od.cpu().detach().numpy()  # .reshape(32, 1125*200)#.reshape(32, 3*200)
            distortion_output = output_feature_l2.cpu().detach().numpy().reshape(256, 64 * 500)  # .reshape(32, 3*200)
            tsne2d = TSNE(n_components=2, init='pca', random_state=0)
            tsne_result_od = tsne2d.fit_transform(output_od)
            tsne_result_d = tsne2d.fit_transform(distortion_output)
            plt.figure(figsize=(20, 6))
            plot = plt.subplot(131)
            plt2 = plot.scatter(tsne_result_od[:, 0], tsne_result_od[:, 1],
                                c=target_od.cpu().detach().numpy())  # c=pred_choice.cpu().detach().numpy()
            plt.title('mlp_od_test', x=0.5, y=-0.1)
            plt.legend(handles=plt2.legend_elements()[0],
                       loc='best',
                       labels=['order', 'disorder'])  # handles=plot.legend_elements()[0],labels=['order', 'disorder']
            plot = plt.subplot(132)
            plt3 = plot.scatter(tsne_result_d[:, 0], tsne_result_d[:, 1],
                                c=target_mol.cpu().detach().numpy(), cmap='rainbow')  # c=pred_choice.cpu().detach().numpy()
            plt.title('conv_mol_test', x=0.5, y=-0.1)
            plt.legend(handles=plt3.legend_elements()[0], loc='best',
                       labels=['1188514',
                               '1286153',
                               '1142500',
                               '248562',
                               '1033994',
                               '1450800',
                               '1249410',
                               '1432562',
                               '2150172',
                               '1100214',
                               '1208853',
                               '1934968'])  # handles=plot.legend_elements()[0],labels=['1188514', '1286153']
            plot = plt.subplot(133)
            plt4 = plot.scatter(tsne_result_d[:, 0], tsne_result_d[:, 1],
                                c=target_od.cpu().detach().numpy())  # c=pred_choice.cpu().detach().numpy()
            plt.title('mlp_od_test', x=0.5, y=-0.1)
            plt.legend(handles=plt4.legend_elements()[0],
                       loc='best',
                       labels=['order', 'disorder'])  # handles=plot.legend_elements()[0],labels=['order', 'disorder']
            # if epoch>17:
            # plt.show()
            plt.savefig(os.path.join(final_output_dir, 'test_Epoch[{0}]_{1}.png'.format(epoch, i)), dpi=720)
            plt.close("all")


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        for cat in np.unique(target_od.cpu()):
            classacc_od = pred_choice_od[target_od == cat].eq(target_od[target_od == cat].long().data).cpu().sum()
            class_od_acc[cat, 0] += classacc_od.item() / float(points[target_od == cat].size()[0])
            class_od_acc[cat, 1] += 1

        correct_od = pred_choice_od.eq(target_od.long().data).cpu().sum()
        mean_correct_od.append(correct_od.item() / float(points.size()[0]))

    class_od_acc[:, 2] = class_od_acc[:, 0] / class_od_acc[:, 1]
    class_od_acc = np.mean(class_od_acc[:, 2])
    print('class_od_acc', class_od_acc)
    instance_od_acc = np.mean(mean_correct_od)
    logging.info(f'instance_od_acc {instance_od_acc:.3f}\t'.format(instance_od_acc=instance_od_acc))
    print('instance_od_acc', instance_od_acc)

    logging.info('=> synchronize...')
    comm.synchronize()

    loss_od_avg, loss_temp_avg, loss_sum_avg= map(
        _meter_reduce if distributed else lambda x: x.avg,
        [losses_od, losses_temp, losses_sum]
    )

    if comm.is_main_process():
        msg = '=> TEST:\t' \
            'Loss_od {loss_od_avg:.4f}\t' \
              'Loss_temp {loss_temp_avg:.4f}\t' \
              'Loss_sum {loss_sum_avg:.4f}\t'.format(
                loss_od_avg=loss_od_avg,
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

    return instance_od_acc, loss_temp_avg

@torch.no_grad()
def test_multi_with_reg_simple_test(final_output_dir, epoch, begin_epoch, end_epoch, config_od, config_mol, val_loader, model, criterion_cls, criterion_reg, output_dir, tb_log_dir,
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
    sysT = []
    predT = []
    targetT = []

    end = time.time()
    for i, (points, target_mol, target_od, target_temp) in enumerate(val_loader):

        points = points.transpose(2, 1)
        # compute output
        points = points.cuda(non_blocking=True)
        target_mol = target_mol.cuda(non_blocking=True)
        target_od = target_od.cuda(non_blocking=True)
        target_temp = target_temp.cuda(non_blocking=True)

        pred_od, pred_temp, Tsys, output_feature_l2, output_feature_l1_od, correlation, c, sa4, resl, resh = model(points)
        pred_choice_od = pred_od.data.max(1)[1]

        # if i == 0:
        #     # tsne_od = output_feature_l1_od.cpu().detach().numpy()
        #     tsne_tod = target_od.cpu().detach().numpy()
        #     tsne_tmol = target_mol.cpu().detach().numpy()
        #     # tsne_dsa4 = sa4.cpu().detach().numpy()
        #     # tsne_d = output_feature_l2.cpu().detach().numpy().reshape(100, 128 * 500)
        #     # tsne_c = correlation.cpu().detach().numpy()
        #     tsne_res = resh.cpu().detach().numpy()
        #     # tsne_dc = c.cpu().detach().numpy()
        #     tsne_dt = Tsys.squeeze(-1).cpu().detach().numpy() - target_temp.cpu().detach().numpy()
        #     # print(target_temp.cpu().detach().numpy().shape, Tsys.cpu().detach().numpy().shape, tsne_dt.shape)
        # else:
        #     # tsne_od = np.vstack((tsne_od, output_feature_l1_od.cpu().detach().numpy()))
        #     # tsne_d = np.vstack((tsne_d, output_feature_l2.cpu().detach().numpy().reshape(100, 128 * 500)))
        #     # tsne_dsa4 = np.vstack((tsne_dsa4, sa4.cpu().detach().numpy()))
        #     tsne_tod = np.hstack((tsne_tod, target_od.cpu().detach().numpy()))
        #     tsne_tmol = np.hstack((tsne_tmol, target_mol.cpu().detach().numpy()))
        #     # tsne_c = np.vstack((tsne_c, correlation.cpu().detach().numpy()))
        #     tsne_res = np.vstack((tsne_res, resh.cpu().detach().numpy()))
        #     # tsne_dc = np.vstack((tsne_dc, c.cpu().detach().numpy()))
        #     tsne_dt = np.hstack((tsne_dt, Tsys.squeeze(-1).cpu().detach().numpy() - target_temp.cpu().detach().numpy()))
        #     # print(tsne_tod.shape, tsne_d.shape, tsne_c.shape, tsne_dt.shape)

        loss_od = criterion_cls(pred_od, target_od.long())
        # print(target_od)
        loss_temp = criterion_reg(pred_temp, target_temp.long())#.sum()
        # print(target_temp)
        # print(pred_temp.mean())
        print(loss_temp)

        loss_sum = loss_od + loss_temp

        losses_od.update(loss_od.item(), points.size(0))
        losses_temp.update(loss_temp.item(), points.size(0))
        losses_sum.update(loss_sum.item(), points.size(0))

        # sysT = np.append(sysT, Tsys.squeeze(-1).cpu().detach().numpy())
        # targetT = np.append(targetT, target_temp.cpu().detach().numpy())
        # predT = np.append(predT, pred_temp.cpu().detach().numpy())

        sysT = np.append(sysT, Tsys.squeeze(-1).cpu().detach().numpy().mean())
        targetT = np.append(targetT, target_temp.cpu().detach().numpy().mean())
        predT = np.append(predT, pred_temp.cpu().detach().numpy().mean())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        for cat in np.unique(target_od.cpu()):
            classacc_od = pred_choice_od[target_od == cat].eq(target_od[target_od == cat].long().data).cpu().sum()
            class_od_acc[cat, 0] += classacc_od.item() / float(points[target_od == cat].size()[0])
            class_od_acc[cat, 1] += 1

        correct_od = pred_choice_od.eq(target_od.long().data).cpu().sum()
        mean_correct_od.append(correct_od.item() / float(points.size()[0]))

    class_od_acc[:, 2] = class_od_acc[:, 0] / class_od_acc[:, 1]
    class_od_acc = np.mean(class_od_acc[:, 2])
    print('class_od_acc', class_od_acc)
    instance_od_acc = np.mean(mean_correct_od)
    logging.info(f'instance_od_acc {instance_od_acc:.3f}\t'.format(instance_od_acc=instance_od_acc))
    print('instance_od_acc', instance_od_acc)

    logging.info('=> synchronize...')
    comm.synchronize()

    # output_od = tsne_od  # .reshape(32, 1125*200)#.reshape(32, 3*200)
    # distortion_output = tsne_dsa4  # .reshape(32, 3*200)
    # correlation_output = tsne_c
    # res_output = tsne_res
    # dc_output = tsne_dc
    # tsne2d = TSNE(n_components=2, init='pca', random_state=0)
    # tsne_result_od = tsne2d.fit_transform(output_od)
    # tsne_result_d = tsne2d.fit_transform(distortion_output)
    # tsne_result_c = tsne2d.fit_transform(correlation_output)
    # tsne_result_dc = tsne2d.fit_transform(dc_output)
    #
    # pca = PCA(n_components=0.95)
    # pca_c = StandardScaler().fit_transform(correlation_output)
    # pca_res = StandardScaler().fit_transform(res_output)
    # pca_dc = StandardScaler().fit_transform(dc_output)
    # pca_distortion = StandardScaler().fit_transform(distortion_output)
    # pca_result_c = pca.fit_transform(pca_c)
    # pca_result_res = pca.fit_transform(pca_res)
    # pca_result_dc = pca.fit_transform(pca_dc)
    # pca_result_d = pca.fit_transform(pca_distortion)
    # # pca_result_mol = pca.fit_transform(pca_mol)

    # plt.figure(figsize=(2, 2))
    # plt.scatter(pca_result_d[:, 0], pca_result_d[:, 1], c=tsne_tmol, cmap='rainbow', s=0.3, lw=0)  # c=pred_choice.cpu().detach().numpy()
    # plt.xlabel('PCA component 1', fontsize=5, va='center')
    # plt.ylabel('PCA component 2', fontsize=5, va='center')
    # plt.title('Global feature PCA', fontsize=5)
    # plt.xticks([-20, -15, -10, -5, 0, 5, 10, 15, 20])
    # plt.yticks([-20, -15, -10, -5, 0, 5, 10, 15, 20])
    # plt.tick_params(axis='both', length=2, pad=2, labelsize=5)
    # # plt.legend(handles=ax.legend_elements()[0], loc=2, labels=['order', 'disorder'], fontsize=5, markerscale=0.3)
    # # plt.axis('square')
    # plt.tight_layout()
    # plt.savefig(os.path.join(final_output_dir, 'pca_global_feature_mol2.png'), dpi=300)
    #
    # plt.figure(figsize=(2, 2))
    # ax = plt.scatter(pca_result_d[:, 0], pca_result_d[:, 1], c=tsne_tod, s=0.3, lw=0)  # c=pred_choice.cpu().detach().numpy()
    # plt.xlabel('PCA component 1', fontsize=5, va='center')
    # plt.ylabel('PCA component 2', fontsize=5, va='center')
    # plt.title('Global feature PCA', fontsize=5)
    # plt.xticks([-20, -15, -10, -5, 0, 5, 10, 15, 20])
    # plt.yticks([-20, -15, -10, -5, 0, 5, 10, 15, 20])
    # plt.tick_params(axis='both', length=2, pad=2, labelsize=5)
    # plt.legend(handles=ax.legend_elements()[0], loc=2, labels=['Crystal', 'Liquid'], fontsize=3, markerscale=0.3, handletextpad=0.1, handlelength=2)
    # # plt.axis('square')
    # plt.tight_layout()
    # plt.savefig(os.path.join(final_output_dir, 'pca_global_feature_od2.png'), dpi=300)
    #
    # plt.figure(figsize=(2, 2))
    # plt.scatter(tsne_result_d[:, 0], tsne_result_d[:, 1], c=tsne_tmol, cmap='rainbow', s=0.3, lw=0)  # c=pred_choice.cpu().detach().numpy()
    # plt.xlabel('t-SNE component 1', fontsize=5, va='center')
    # plt.ylabel('t-SNE component 2', fontsize=5, va='center')
    # plt.title('Global feature t-SNE', fontsize=5)
    # # plt.xticks([-20, -15, -10, -5, 0, 5, 10, 15, 20])
    # # plt.yticks([-20, -15, -10, -5, 0, 5, 10, 15, 20])
    # plt.tick_params(axis='both', length=2, pad=2, labelsize=5)
    # # plt.legend(handles=ax.legend_elements()[0], loc=2, labels=['order', 'disorder'], fontsize=5, markerscale=0.3)
    # # plt.axis('square')
    # plt.tight_layout()
    # plt.savefig(os.path.join(final_output_dir, 'tsne_global_feature_mol_valid2.png'), dpi=300)
    #
    # plt.figure(figsize=(2, 2))
    # ax = plt.scatter(tsne_result_d[:, 0], tsne_result_d[:, 1], c=tsne_tod, s=0.3, lw=0)  # c=pred_choice.cpu().detach().numpy()
    # plt.xlabel('t-SNE component 1', fontsize=5, va='center')
    # plt.ylabel('t-SNE component 2', fontsize=5, va='center')
    # plt.title('Global feature t-SNE', fontsize=5)
    # # plt.xticks([-20, -15, -10, -5, 0, 5, 10, 15, 20])
    # # plt.yticks([-20, -15, -10, -5, 0, 5, 10, 15, 20])
    # plt.tick_params(axis='both', length=2, pad=2, labelsize=5)
    # plt.legend(handles=ax.legend_elements()[0], loc=2, labels=['Crystal', 'Liquid'], fontsize=3, markerscale=0.3, handletextpad=0.1, handlelength=2)
    # # plt.axis('square')
    # plt.tight_layout()
    # plt.savefig(os.path.join(final_output_dir, 'tsne_global_feature_od_valid2.png'), dpi=300)

    # plt.figure(figsize=(2, 2))
    # ax = plt.scatter(pca_result_c[:, 0], pca_result_c[:, 1], s=0.3, lw=0, c=tsne_dt, cmap='plasma', vmin=-60, vmax=60)  # c=pred_choice.cpu().detach().numpy()
    # plt.xlabel('PCA component 1', fontsize=5, va='center')
    # plt.ylabel('PCA component 2', fontsize=5, va='center')
    # plt.title('Correlation PCA', fontsize=5)
    # plt.tick_params(axis='both', length=2, pad=2, labelsize=5)
    # # plt.xlim([-20, 35])
    # # plt.xticks([-20, -10, 0, 10, 20, 30])
    # # plt.ylim([-5, 15])
    # # plt.yticks([-5, 0, 5, 10, 15])
    # clb = plt.colorbar(ax, ticks=[-50, -40, -30, -20, -10, 10, 20, 30, 40, 50])
    # # plt.legend(handles=ax.legend_elements()[0], loc=2, labels=['order', 'disorder'], fontsize=5, markerscale=0.3)
    # clb.set_label('TBMP (K)', fontsize=5, rotation=270, labelpad=8.5)
    # clb.ax.tick_params(labelsize=5)
    # # plt.axis('equal')
    # plt.tight_layout()
    # plt.savefig(os.path.join(final_output_dir, 'pca_correlation_temp2.png'), dpi=300)

    # plt.figure(figsize=(2.4, 2))
    # ax = plt.scatter(pca_result_res[:, 0], pca_result_res[:, 1], s=0.3, lw=0, c=tsne_dt, cmap='plasma', vmin=0, vmax=60)  # c=pred_choice.cpu().detach().numpy()
    # plt.xlabel('PCA component 1', fontsize=5, va='center')
    # plt.ylabel('PCA component 2', fontsize=5, va='center')
    # plt.title('Extended feature PCA', fontsize=5)
    # plt.tick_params(axis='both', length=2, pad=2, labelsize=5)
    # plt.xlim([-20, 35])
    # plt.xticks([-20, -10, 0, 10, 20, 30])
    # plt.ylim([-5, 15])
    # plt.yticks([-5, 0, 5, 10, 15])
    # clb = plt.colorbar(ax, ticks=[10, 20, 30, 40, 50])
    # # plt.legend(handles=ax.legend_elements()[0], loc=2, labels=['order', 'disorder'], fontsize=5, markerscale=0.3)
    # clb.set_label('TBMP (K)', fontsize=5, rotation=270, labelpad=8.5)
    # clb.ax.tick_params(labelsize=5)
    # # plt.axis('equal')
    # plt.tight_layout()
    # plt.savefig(os.path.join(final_output_dir, 'pca_total_temp_liquid2.png'), dpi=300)

    # plt.figure(figsize=(2.4, 2))
    # ax = plt.scatter(pca_result_res[:, 0], pca_result_res[:, 1], s=0.3, lw=0, c=tsne_dt, cmap='plasma', vmin=-60, vmax=0)  # c=pred_choice.cpu().detach().numpy()
    # plt.xlabel('PCA component 1', fontsize=5, va='center')
    # plt.ylabel('PCA component 2', fontsize=5, va='center')
    # plt.title('Extended feature PCA', fontsize=5)
    # plt.tick_params(axis='both', length=2, pad=2, labelsize=5)
    # plt.xlim([-30, 35])
    # plt.xticks([-25, -15, -5, 5, 15, 25])
    # plt.ylim([-5, 15])
    # plt.yticks([-5, 0, 5, 10, 15])
    # clb = plt.colorbar(ax, ticks=[-50, -40, -30, -20, -10])
    # # plt.legend(handles=ax.legend_elements()[0], loc=2, labels=['order', 'disorder'], fontsize=5, markerscale=0.3)
    # clb.set_label('TBMP (K)', fontsize=5, rotation=270, labelpad=8.5)
    # clb.ax.tick_params(labelsize=5)
    # # plt.axis('equal')
    # plt.tight_layout()
    # plt.savefig(os.path.join(final_output_dir, 'pca_total_temp_solid2.png'), dpi=300)

    # plt.figure(figsize=(48, 6))
    # # plt.rcParams['font.sans-serif'] = ['Times New Roman']
    # plot = plt.subplot(171)
    # plot.set_aspect('equal', adjustable='box')
    # plt2 = plot.scatter(tsne_result_od[:, 0], tsne_result_od[:, 1],
    #                     c=tsne_tod, s=20)  # c=pred_choice.cpu().detach().numpy()
    # plt.title('mlp_od_test', x=0.5, y=-0.1)
    # plt.legend(handles=plt2.legend_elements()[0],
    #            loc='best',
    #            labels=['order', 'disorder'])  # handles=plot.legend_elements()[0],labels=['order', 'disorder']
    # plot = plt.subplot(172)
    # plot.set_aspect('equal', adjustable='box')
    # plt3 = plot.scatter(tsne_result_d[:, 0], tsne_result_d[:, 1],
    #                     c=tsne_tmol, cmap='rainbow', s=20)  # c=pred_choice.cpu().detach().numpy()
    # plt.title('conv_mol_test', x=0.5, y=-0.1)
    # # plt.legend(handles=plt3.legend_elements()[0],
    # #            loc='best')  # handles=plot.legend_elements()[0],labels=['1188514', '1286153']
    # plot = plt.subplot(173)
    # plot.set_aspect('equal', adjustable='box')
    # plt4 = plot.scatter(tsne_result_d[:, 0], tsne_result_d[:, 1],
    #                     c=tsne_tod, s=20)  # c=pred_choice.cpu().detach().numpy()
    # plt.title('conv_od_test', x=0.5, y=-0.1)
    # plt.legend(handles=plt4.legend_elements()[0],
    #            loc='best',
    #            labels=['order', 'disorder'])  # handles=plot.legend_elements()[0],labels=['order', 'disorder']
    # plot = plt.subplot(174)
    # plot.set_aspect('equal', adjustable='box')
    # plt5 = plot.scatter(tsne_result_c[:, 0], tsne_result_c[:, 1],
    #                     c=tsne_dt, cmap='plasma', vmin=-60, vmax=60, s=20)
    # plt.colorbar(plt5, ticks=[-50, -40, -30, -20, -10, 10, 20, 30, 40, 50],
    #              shrink=0.8)  # c=pred_choice.cpu().detach().numpy()
    # # plt.legend(handles=plt5.legend_elements()[0],
    # #            loc='best')
    # plt.title('correlation', x=0.5, y=-0.2)
    # plot = plt.subplot(175)
    # plot.set_aspect('equal', adjustable='box')
    # plt6 = plot.scatter(tsne_result_dc[:, 0], tsne_result_dc[:, 1],
    #                     c=tsne_dt, cmap='plasma', vmin=-60, vmax=60, s=20)  # c=pred_choice.cpu().detach().numpy()
    # plt.colorbar(plt6, ticks=[-50, -40, -30, -20, -10, 10, 20, 30, 40, 50], shrink=0.8)
    # # plt.legend(handles=plt6.legend_elements()[0],
    # #            loc='best')
    # plt.title('correlation', x=0.5, y=-0.2)
    # plot = plt.subplot(176)
    # plot.set_aspect('equal', adjustable='box')
    # plt7 = plot.scatter(pca_result_c[:, 0], pca_result_c[:, 1],
    #                     c=tsne_dt, cmap='plasma', vmin=-60, vmax=60, s=1)
    # plt.colorbar(plt7, ticks=[-50, -40, -30, -20, -10, 10, 20, 30, 40, 50])  # c=pred_choice.cpu().detach().numpy()
    # # plt.legend(handles=plt5.legend_elements()[0],
    # #            loc='best')
    # plt.title('correlation', x=0.5, y=-0.1)
    # plot = plt.subplot(177)
    # plot.set_aspect('equal', adjustable='box')
    # plt8 = plot.scatter(pca_result_dc[:, 0], pca_result_dc[:, 1],
    #                     c=tsne_dt, cmap='plasma', vmin=-60, vmax=60, s=1)  # c=pred_choice.cpu().detach().numpy()
    # plt.colorbar(plt8, ticks=[-50, -40, -30, -20, -10, 10, 20, 30, 40, 50], shrink=0.5)
    # # plt.legend(handles=plt6.legend_elements()[0],
    # #            loc='best')
    # plt.title('correlation', x=0.5, y=-0.2)
    # # plot = plt.subplot(254)
    # # plot.set_aspect('equal', adjustable='box')
    # # plt9 = plot.scatter(pca_result_mol[:, 0], pca_result_mol[:, 1],
    # #                     c=tsne_tmol, cmap='rainbow')  # c=pred_choice.cpu().detach().numpy()
    # # plt.title('mlp_mol_test', x=0.5, y=-0.1)
    # # plot = plt.subplot(255)
    # # plot.set_aspect('equal', adjustable='box')
    # # plt10 = plot.scatter(pca_result_d[:, 0], pca_result_d[:, 1],
    # #                     c=tsne_tmol, cmap='rainbow')  # c=pred_choice.cpu().detach().numpy()
    # # plt.title('conv_mol_test', x=0.5, y=-0.1)
    #
    # # if epoch>17:
    # # plt.show()
    # plt.savefig(os.path.join(final_output_dir, 'test_Epoch[{0}]_{1}.png'), dpi=300)
    # plt.close("all")

    # plt.figure(figsize=(18, 6))
    # plot = plt.subplot(121)
    # plt1 = plot.scatter(pca_result_d[:, 0], pca_result_d[:, 1],
    #                     c=tsne_tmol, cmap='rainbow')  # c=pred_choice.cpu().detach().numpy()
    # plt.title('conv_mol_test', x=0.5, y=-0.1)
    # plot = plt.subplot(122)
    # plt2 = plot.scatter(pca_result_d[:, 0], pca_result_d[:, 1],
    #                     c=tsne_tod)  # c=pred_choice.cpu().detach().numpy()
    # plt.title('conv_od_test', x=0.5, y=-0.1)
    # plt.legend(handles=plt2.legend_elements()[0],
    #                       loc='best',
    #                       labels=['order', 'disorder'])  # handles=plot.legend_elements()[0],labels=['order', 'disorder']
    # plt.savefig(os.path.join(final_output_dir, 'pca.png'), dpi=300)

    loss_od_avg, loss_mol_avg, loss_temp_avg, loss_sum_avg= map(
        _meter_reduce if distributed else lambda x: x.avg,
        [losses_od, losses_mol, losses_temp, losses_sum]
    )

    sysT = np.array(sysT).reshape(-1)
    targetT = np.array(targetT).reshape(-1)
    predT = np.array(predT).reshape(-1)
    # #
    # deltaT_pred = sysT - predT
    # deltaT_real = sysT - targetT
    # #
    # mae = mean_absolute_error(deltaT_real, deltaT_pred)
    # rmse = np.sqrt(mean_squared_error(deltaT_real, deltaT_pred))
    # r_2 = r2_score(deltaT_real, deltaT_pred)

    mae = mean_absolute_error(predT, targetT)
    rmse = np.sqrt(mean_squared_error(predT, targetT))
    r_2 = r2_score(predT, targetT)

    # x45 = np.linspace(-50.0, 50.0, 100)
    # y45=x45
    #
    # plt.figure(figsize=(2, 2))
    # plt.plot(x45, y45, c='r', linewidth=0.5)
    # plt.scatter(deltaT_real, deltaT_pred, s=1)  # c=pred_choice.cpu().detach().numpy()
    # plt.text(-53, 50, s=f"$R^2$={round(r_2, 2)}", fontsize=5)
    # plt.text(-53, 42, 'rmse={:.2f}'.format(rmse), fontsize=5)
    # plt.text(-53, 34, 'mae={:.2f}'.format(mae), fontsize=5)
    # # plt.axvline(-47, color='black', linestyle='--') #shu
    # # plt.axvline(-37, color='black', linestyle='--')
    # # plt.axvline(-27, color='black', linestyle='--')
    # # plt.axvline(-17, color='black', linestyle='--')
    # # plt.axvline(13, color='black', linestyle='--')
    # # plt.axvline(23, color='black', linestyle='--')
    # # plt.axvline(33, color='black', linestyle='--')
    # # plt.axvline(43, color='black', linestyle='--')
    # # plt.axhline(-47, color='black', linestyle='--') #hen
    # # plt.axhline(-37, color='black', linestyle='--')
    # # plt.axhline(-27, color='black', linestyle='--')
    # # plt.axhline(-17, color='black', linestyle='--')
    # # plt.axhline(13, color='black', linestyle='--') #hen
    # # plt.axhline(23, color='black', linestyle='--')
    # # plt.axhline(33, color='black', linestyle='--')
    # # plt.axhline(43, color='black', linestyle='--')
    # plt.xlim([-50.0, 50.0])
    # plt.xticks([-40,-20,0,20,40])
    # plt.ylim([-50.0, 50.0])
    # plt.yticks([-40,-20,0,20,40])
    # plt.tick_params(axis='both', length=2, pad=2, labelsize=5)
    # plt.axis('square')
    # plt.xlabel('True TBMP (K)', fontsize=5, va='center')
    # plt.ylabel('Predicted TBMP (K)', fontsize=5, va='center')
    # plt.tight_layout()
    #
    # plt.savefig(os.path.join(final_output_dir, 'test_Epoch_rmse_113810.png'), dpi=300)

    # x45 = np.linspace(0, 60.0, 100)
    # y45=x45
    #
    # plt.figure(figsize=(2, 2))
    # plt.scatter(deltaT_real, deltaT_pred, s=1)  # c=pred_choice.cpu().detach().numpy()
    # plt.text(-1, 61, s=f"$R^2$={round(r_2, 2)}", fontsize=5)
    # plt.text(-1, 56, 'rmse={:.2f}'.format(rmse), fontsize=5)
    # plt.text(-1, 51, 'mae={:.2f}'.format(mae), fontsize=5)
    # plt.plot(x45, y45, c='r', linewidth=0.5)
    # plt.xlim([0, 60.0])
    # plt.xticks([0,20,40,60])
    # plt.ylim([0, 60.0])
    # plt.yticks([0,20,40,60])
    # plt.tick_params(axis='both', length=2, pad=2, labelsize=5)
    # plt.axis('square')
    # plt.xlabel('True TBMP (K)', fontsize=5, va='center')
    # plt.ylabel('Predicted TBMP (K)', fontsize=5, va='center')
    # plt.tight_layout()
    #
    # plt.savefig(os.path.join(final_output_dir, 'valid.png'), dpi=300)
    # plt.close("all")

    # x45 = np.linspace(-100.0, 100.0, 200)
    # y45=x45
    #
    # plt.figure(figsize=(6, 6))
    # plt.scatter(deltaT_real, deltaT_pred, c='r', s=1)  # c=pred_choice.cpu().detach().numpy()
    # plt.text( -65, 65, 'R2={:.2f}'.format(r_2))
    # plt.text( -65, 60, 'rmse={:.2f}'.format(rmse))
    # plt.text( -65, 55, 'mae={:.2f}'.format(mae))
    # plt.plot(x45,y45)
    # # plt.axvline(-47, color='black', linestyle='--') #shu
    # # plt.axvline(-37, color='black', linestyle='--')
    # # plt.axvline(-27, color='black', linestyle='--')
    # # plt.axvline(-17, color='black', linestyle='--')
    # # plt.axvline(13, color='black', linestyle='--')
    # # plt.axvline(23, color='black', linestyle='--')
    # # plt.axvline(33, color='black', linestyle='--')
    # # plt.axvline(43, color='black', linestyle='--')
    # # plt.axhline(-47, color='black', linestyle='--') #hen
    # # plt.axhline(-37, color='black', linestyle='--')
    # # plt.axhline(-27, color='black', linestyle='--')
    # # plt.axhline(-17, color='black', linestyle='--')
    # # plt.axhline(13, color='black', linestyle='--') #hen
    # # plt.axhline(23, color='black', linestyle='--')
    # # plt.axhline(33, color='black', linestyle='--')
    # # plt.axhline(43, color='black', linestyle='--')
    # plt.xlim([-70.0, 70.0])
    # plt.ylim([-70.0, 70.0])
    # plt.xlabel('deltaT_real')
    # plt.ylabel('deltaT_predict')
    #
    # plt.savefig(os.path.join(final_output_dir, 'test_Epoch_rmse_113810.png'), dpi=300)

    # x45 = np.linspace(-100.0, 100.0, 200)
    # y45=x45
    #
    # plt.figure(figsize=(6, 6))
    # plt.scatter(deltaT_real, deltaT_pred, c='r', s=1)  # c=pred_choice.cpu().detach().numpy()
    # plt.text( 5, 55, 'R2={:.2f}'.format(r_2))
    # plt.text( 5, 50, 'rmse={:.2f}'.format(rmse))
    # plt.text( 5, 45, 'mae={:.2f}'.format(mae))
    # plt.plot(x45,y45)
    # plt.axvline(-47, color='black', linestyle='--') #shu
    # plt.axvline(-37, color='black', linestyle='--')
    # plt.axvline(-27, color='black', linestyle='--')
    # plt.axvline(-17, color='black', linestyle='--')
    # plt.axvline(13, color='black', linestyle='--')
    # plt.axvline(23, color='black', linestyle='--')
    # plt.axvline(33, color='black', linestyle='--')
    # plt.axvline(43, color='black', linestyle='--')
    # plt.axhline(-47, color='black', linestyle='--') #hen
    # plt.axhline(-37, color='black', linestyle='--')
    # plt.axhline(-27, color='black', linestyle='--')
    # plt.axhline(-17, color='black', linestyle='--')
    # plt.axhline(13, color='black', linestyle='--') #hen
    # plt.axhline(23, color='black', linestyle='--')
    # plt.axhline(33, color='black', linestyle='--')
    # plt.axhline(43, color='black', linestyle='--')
    # plt.xlim([0, 60.0])
    # plt.ylim([0, 60.0])
    # plt.xlabel('real TBMP')
    # plt.ylabel('predicted TBMP')
    #
    # plt.savefig(os.path.join(final_output_dir, 'valid.png'), dpi=300)

    x45 = np.linspace(280.0, 460.0, 180)
    y45=x45
    Ttest = targetT[0:76] #s76l70
    TPtest = predT[0:76]
    print(TPtest.shape)
    Tvalid = targetT[76:]
    TPvalid = predT[76:]
    print(TPvalid.shape)

    # maev = mean_absolute_error(TPtest, Ttest)
    # rmsev = np.sqrt(mean_squared_error(TPtest, Ttest))
    # r_2v = r2_score(TPtest, Ttest)
    # print("validation", rmsev, maev, r_2v)
    #
    # maet = mean_absolute_error(TPvalid, Tvalid)
    # rmset = np.sqrt(mean_squared_error(TPvalid, Tvalid))
    # r_2t = r2_score(TPvalid, Tvalid)
    # print("test", rmset, maet, r_2t)

    plt.figure(figsize=(2, 2))
    plt.scatter(Ttest, TPtest , s=1, marker='s', label='Validation')  # c=pred_choice.cpu().detach().numpy()
    plt.scatter(Tvalid, TPvalid, c='orange', s=1, marker='^', label='Test')  # c=pred_choice.cpu().detach().numpy()
    plt.text(275, 458, s=f"$R^2$={round(r_2, 2)}", fontsize=5)
    plt.text(275, 445, 'RMSE={:.2f}'.format(rmse), fontsize=5)
    plt.text(275, 431, 'MAE={:.2f}'.format(mae), fontsize=5)
    plt.plot(x45, y45, c='r', linewidth=0.5)
    plt.xlim([280, 460])
    plt.xticks(np.arange(280,461,30))
    plt.ylim([280, 460])
    plt.yticks(np.arange(280,461,30))
    plt.tick_params(axis='both', length=2, pad=2, labelsize=5)
    plt.axis('square')
    plt.xlabel('True MP (K)', fontsize=5, va='center')
    plt.ylabel('Predicted MP (K)', fontsize=5, va='center')
    plt.legend(loc=4, fontsize=5)
    plt.tight_layout()

    plt.savefig(os.path.join(final_output_dir, 'test_rmse_solid2.png'), dpi=300)
    plt.close("all")

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

    return instance_od_acc, loss_temp_avg


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
        points = points.cuda(non_blocking=True)
        target_od = target_od.cuda(non_blocking=True)
        target_mol = target_mol.cuda(non_blocking=True)
        target_temp = target_temp.cuda(non_blocking=True)

        pred_temp, Tsys, output_feature_l2,correlation = model(points)

        loss_temp = criterion_reg(pred_temp, MaxMinNormalization((Tsys - target_temp.to(torch.float32)), 0, 70))#.sum()
        print(UnMaxMinNormalization(pred_temp, 0, 70) - (Tsys - target_temp))
        print(loss_temp)

        pred_temp = UnMaxMinNormalization(pred_temp, 0, 70)

        sysT = np.append(sysT, Tsys.squeeze(-1).cpu().detach().numpy())
        targetT = np.append(targetT, target_temp.cpu().detach().numpy())
        predT = np.append(predT, pred_temp.cpu().detach().numpy())

        losses_temp.update(loss_temp.item(), points.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logging.info('=> synchronize...')
    comm.synchronize()

    sysT = np.array(sysT).reshape(-1)
    targetT = np.array(targetT).reshape(-1)
    predT = np.array(predT).reshape(-1)

    deltaT_pred = predT
    deltaT_real = sysT - targetT

    # loss_temp_avg= losses_temp.avg
    r_2 = r2_score(deltaT_real, deltaT_pred)
    loss_temp_avg = -r_2

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

def test_multi_with_reg_simpler_test(final_output_dir, epoch, begin_epoch, end_epoch, config_od, config_mol, val_loader, model, criterion_cls, criterion_reg, output_dir, tb_log_dir,
         writer_dict=None, distributed=False): #final_output_dir, epoch,
    batch_time = AverageMeter()
    losses_temp = AverageMeter()
    #top1 = AverageMeter()
    #top2 = AverageMeter()

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
        # target_od = target_od.cuda(non_blocking=True)
        # target_mol = target_mol.cuda(non_blocking=True)
        target_temp = target_temp.cuda(non_blocking=True)

        pred_temp, Tsys, output_feature_l2,  correlation = model(points) #, c, sa4, resl

        # if i == 0:
        #     tsne_tod = target_od.cpu().detach().numpy()
        #     tsne_tmol = target_mol.cpu().detach().numpy()
        #     # tsne_d = output_feature_l2.cpu().detach().numpy().reshape(100, 128 * 500)
        #     # tsne_dsa4 = sa4.cpu().detach().numpy()
        #     tsne_c = correlation.cpu().detach().numpy()
        #     # tsne_dc = c.cpu().detach().numpy()
        #     tsne_res = resl.cpu().detach().numpy()
        #     tsne_dt = Tsys.squeeze(-1).cpu().detach().numpy() - target_temp.cpu().detach().numpy()
        #     # print(target_temp.cpu().detach().numpy().shape, Tsys.cpu().detach().numpy().shape, tsne_dt.shape)
        # else:
        #     # tsne_d = np.vstack((tsne_d, output_feature_l2.cpu().detach().numpy().reshape(100, 128 * 500)))
        #     # tsne_dsa4 = np.vstack((tsne_dsa4, sa4.cpu().detach().numpy()))
        #     tsne_tod = np.hstack((tsne_tod, target_od.cpu().detach().numpy()))
        #     tsne_tmol = np.hstack((tsne_tmol, target_mol.cpu().detach().numpy()))
        #     tsne_c = np.vstack((tsne_c, correlation.cpu().detach().numpy()))
        #     # tsne_dc = np.vstack((tsne_dc, c.cpu().detach().numpy()))
        #     tsne_res = np.vstack((tsne_res, resl.cpu().detach().numpy()))
        #     tsne_dt = np.hstack((tsne_dt, Tsys.squeeze(-1).cpu().detach().numpy() - target_temp.cpu().detach().numpy()))
        #     # print(tsne_tod.shape, tsne_d.shape, tsne_c.shape, tsne_dt.shape)

        loss_temp = criterion_reg(pred_temp, MaxMinNormalization((Tsys - target_temp.to(torch.float32)), -70, 70))#.sum()
        # print(pred_temp)
        print(loss_temp)

        pred_temp = UnMaxMinNormalization(pred_temp, -70, 70)

        losses_temp.update(loss_temp.item(), points.size(0))

        sysT = np.append(sysT, Tsys.squeeze(-1).cpu().detach().numpy())
        targetT = np.append(targetT, target_temp.cpu().detach().numpy())
        predT = np.append(predT, pred_temp.cpu().detach().numpy())

        # sysT = np.append(sysT, Tsys.squeeze(-1).cpu().detach().numpy().mean())
        # targetT = np.append(targetT, target_temp.cpu().detach().numpy().mean())
        # predT = np.append(predT, pred_temp.cpu().detach().numpy().mean())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logging.info('=> synchronize...')
    comm.synchronize()

    print(np.mean(predT))

    # distortion_output = tsne_dsa4 # .reshape(32, 3*200)
    # correlation_output = tsne_c
    # dc_output = tsne_dc
    # res_output = tsne_res
    # tsne2d = TSNE(n_components=2, init='pca', random_state=0)#random_state=0
    # tsne_result_d = tsne2d.fit_transform(distortion_output)
    # tsne_result_c = tsne2d.fit_transform(correlation_output)
    # tsne_result_dc = tsne2d.fit_transform(dc_output)
    #
    # pca = PCA(n_components=0.95)
    # pca_c = StandardScaler().fit_transform(correlation_output)
    # pca_dc = StandardScaler().fit_transform(dc_output)
    # pca_res = StandardScaler().fit_transform(res_output)
    # pca_distortion = StandardScaler().fit_transform(distortion_output)
    # pca_result_c = pca.fit_transform(pca_c)
    # pca_result_dc = pca.fit_transform(pca_dc)
    # pca_result_res = pca.fit_transform(pca_res)
    # pca_result_d = pca.fit_transform(pca_distortion)
    # pca_result_mol = pca.fit_transform(pca_mol)

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.serif'] = ['Arial']

    # plt.figure(figsize=(2, 2))
    # plt.scatter(tsne_result_d[:, 0], tsne_result_d[:, 1], c=tsne_tmol, cmap='rainbow', s=0.3, lw=0)  # c=pred_choice.cpu().detach().numpy()
    # plt.xlabel('t-SNE component 1', fontsize=5, va='center')
    # plt.ylabel('t-SNE component 2', fontsize=5, va='center')
    # plt.title('Global feature t-SNE', fontsize=5)
    # # plt.xticks([-20, -15, -10, -5, 0, 5, 10, 15, 20])
    # # plt.yticks([-20, -15, -10, -5, 0, 5, 10, 15, 20])
    # plt.tick_params(axis='both', length=2, pad=2, labelsize=5)
    # # plt.legend(handles=ax.legend_elements()[0], loc=2, labels=['order', 'disorder'], fontsize=5, markerscale=0.3)
    # # plt.axis('square')
    # plt.tight_layout()
    # plt.savefig(os.path.join(final_output_dir, 'tsne_global_feature_mol_valid2.png'), dpi=300)

    # plt.figure(figsize=(2, 2))
    # ax = plt.scatter(tsne_result_d[:, 0], tsne_result_d[:, 1], c=tsne_tod, s=0.3, lw=0)  # c=pred_choice.cpu().detach().numpy()
    # plt.xlabel('t-SNE component 1', fontsize=5, va='center')
    # plt.ylabel('t-SNE component 2', fontsize=5, va='center')
    # plt.title('Global feature t-SNE', fontsize=5)
    # # plt.xticks([-20, -15, -10, -5, 0, 5, 10, 15, 20])
    # # plt.yticks([-20, -15, -10, -5, 0, 5, 10, 15, 20])
    # plt.tick_params(axis='both', length=2, pad=2, labelsize=5)
    # plt.legend(handles=ax.legend_elements()[0], loc=2, labels=['Crystal', 'Liquid'], fontsize=3, markerscale=0.3, handletextpad=0.1, handlelength=2)
    # # plt.axis('square')
    # plt.tight_layout()
    # plt.savefig(os.path.join(final_output_dir, 'tsne_global_feature_od_valid2.png'), dpi=300)

    # plt.figure(figsize=(2, 2))
    # ax = plt.scatter(pca_result_c[:, 0], pca_result_c[:, 1], s=0.3, lw=0, c=tsne_dt, cmap='plasma', vmin=-60, vmax=60)  # c=pred_choice.cpu().detach().numpy()
    # plt.xlabel('PCA component 1', fontsize=5, va='center')
    # plt.ylabel('PCA component 2', fontsize=5, va='center')
    # plt.title('Correlation PCA', fontsize=5)
    # plt.tick_params(axis='both', length=2, pad=2, labelsize=5)
    # # plt.xlim([-15, 15])
    # # plt.xticks([-15, -10, -5, 0, 5, 10, 15])
    # # plt.ylim([-10, 15])
    # # plt.yticks([-10, -5, 0, 5, 10, 15])
    # clb = plt.colorbar(ax, ticks=[-50, -40, -30, -20, -10, 10, 20, 30, 40, 50])
    # # plt.legend(handles=ax.legend_elements()[0], loc=2, labels=['order', 'disorder'], fontsize=5, markerscale=0.3)
    # clb.set_label('TBMP (K)', fontsize=5, rotation=270, labelpad=8.5)
    # clb.ax.tick_params(labelsize=5)
    # # plt.axis('square')
    # plt.tight_layout()
    # plt.savefig(os.path.join(final_output_dir, 'pca_correlation_temp_valid2.png'), dpi=300)
    #
    # plt.figure(figsize=(2.4, 2))
    # ax = plt.scatter(pca_result_res[:, 0], pca_result_res[:, 1], s=0.3, lw=0, c=tsne_dt, cmap='plasma', vmin=-60, vmax=60)  # c=pred_choice.cpu().detach().numpy()
    # plt.xlabel('PCA component 1', fontsize=5, va='center')
    # plt.ylabel('PCA component 2', fontsize=5, va='center')
    # plt.title('Extended feature PCA', fontsize=5)
    # plt.tick_params(axis='both', length=2, pad=2, labelsize=5)
    # plt.xlim([-15, 15])
    # plt.xticks([-15, -10, -5, 0, 5, 10, 15])
    # plt.ylim([-10, 15])
    # plt.yticks([-10, -5, 0, 5, 10, 15])
    # clb = plt.colorbar(ax, ticks=[-50, -40, -30, -20, -10, 10, 20, 30, 40, 50])
    # # plt.legend(handles=ax.legend_elements()[0], loc=2, labels=['order', 'disorder'], fontsize=5, markerscale=0.3)
    # clb.set_label('TBMP (K)', fontsize=5, rotation=270, labelpad=8.5)
    # clb.ax.tick_params(labelsize=5)
    # # plt.axis('square')
    # plt.tight_layout()
    # plt.savefig(os.path.join(final_output_dir, 'pca_total_temp_res_valid2.png'), dpi=300)


    # plt.figure(figsize=(42, 6))
    # plt.rcParams['font.sans-serif'] = ['Times New Roman']
    # plot = plt.subplot(171)
    # plot.set_aspect('equal', adjustable='box')
    # plt2 = plot.scatter(tsne_result_od[:, 0], tsne_result_od[:, 1],
    #                     c=tsne_tod, s=20)  # c=pred_choice.cpu().detach().numpy()
    # plt.title('mlp_od_test', x=0.5, y=-0.1)
    # plt.legend(handles=plt2.legend_elements()[0],
    #            loc='best',
    #            labels=['order', 'disorder'])  # handles=plot.legend_elements()[0],labels=['order', 'disorder']
    # plot = plt.subplot(161)
    # plot.set_aspect('equal', adjustable='box')
    # plt3 = plot.scatter(tsne_result_d[:, 0], tsne_result_d[:, 1],
    #                     c=tsne_tmol, cmap='rainbow', s=20)  # c=pred_choice.cpu().detach().numpy()
    # plt.title('conv_mol_test', x=0.5, y=-0.1)
    # # plt.legend(handles=plt3.legend_elements()[0],
    # #            loc='best')  # handles=plot.legend_elements()[0],labels=['1188514', '1286153']
    # plot = plt.subplot(162)
    # plot.set_aspect('equal', adjustable='box')
    # plt4 = plot.scatter(tsne_result_d[:, 0], tsne_result_d[:, 1],
    #                     c=tsne_tod, s=20)  # c=pred_choice.cpu().detach().numpy()
    # plt.title('conv_od_test', x=0.5, y=-0.1)
    # plt.legend(handles=plt4.legend_elements()[0],
    #            loc='best',
    #            labels=['order', 'disorder'])  # handles=plot.legend_elements()[0],labels=['order', 'disorder']
    # plot = plt.subplot(163)
    # plot.set_aspect('equal', adjustable='box')
    # plt5 = plot.scatter(tsne_result_c[:, 0], tsne_result_c[:, 1],
    #                     c=tsne_dt, cmap='plasma', vmin=-60, vmax=60, s=20)
    # plt.colorbar(plt5, ticks=[-50, -40, -30, -20, -10, 10, 20, 30, 40, 50],
    #              shrink=0.8)  # c=pred_choice.cpu().detach().numpy()
    # # plt.legend(handles=plt5.legend_elements()[0],
    # #            loc='best')
    # plt.title('correlation', x=0.5, y=-0.2)
    # plot = plt.subplot(164)
    # plot.set_aspect('equal', adjustable='box')
    # plt6 = plot.scatter(tsne_result_dc[:, 0], tsne_result_dc[:, 1],
    #                     c=tsne_dt, cmap='plasma', vmin=-60, vmax=60, s=20)  # c=pred_choice.cpu().detach().numpy()
    # plt.colorbar(plt6, ticks=[-50, -40, -30, -20, -10, 10, 20, 30, 40, 50], shrink=0.8)
    # # plt.legend(handles=plt6.legend_elements()[0],
    # #            loc='best')
    # plt.title('correlation', x=0.5, y=-0.2)
    # plot = plt.subplot(165)
    # plot.set_aspect('equal', adjustable='box')
    # plt7 = plot.scatter(pca_result_c[:, 0], pca_result_c[:, 1],
    #                     c=tsne_dt, cmap='plasma', vmin=-60, vmax=60, s=1)
    # plt.colorbar(plt7, ticks=[-50, -40, -30, -20, -10, 10, 20, 30, 40, 50])  # c=pred_choice.cpu().detach().numpy()
    # # plt.legend(handles=plt5.legend_elements()[0],
    # #            loc='best')
    # plt.title('correlation', x=0.5, y=-0.1)
    # plot = plt.subplot(166)
    # plot.set_aspect('equal', adjustable='box')
    # plt8 = plot.scatter(pca_result_dc[:, 0], pca_result_dc[:, 1],
    #                     c=tsne_dt, cmap='plasma', vmin=-60, vmax=60, s=1)  # c=pred_choice.cpu().detach().numpy()
    # plt.colorbar(plt8, ticks=[-50, -40, -30, -20, -10, 10, 20, 30, 40, 50], shrink=0.5)
    # # plt.legend(handles=plt6.legend_elements()[0],
    # #            loc='best')
    # plt.title('correlation', x=0.5, y=-0.2)
    # # plot = plt.subplot(254)
    # # plot.set_aspect('equal', adjustable='box')
    # # plt9 = plot.scatter(pca_result_mol[:, 0], pca_result_mol[:, 1],
    # #                     c=tsne_tmol, cmap='rainbow')  # c=pred_choice.cpu().detach().numpy()
    # # plt.title('mlp_mol_test', x=0.5, y=-0.1)
    # # plot = plt.subplot(255)
    # # plot.set_aspect('equal', adjustable='box')
    # # plt10 = plot.scatter(pca_result_d[:, 0], pca_result_d[:, 1],
    # #                     c=tsne_tmol, cmap='rainbow')  # c=pred_choice.cpu().detach().numpy()
    # # plt.title('conv_mol_test', x=0.5, y=-0.1)
    #
    # # if epoch>17:
    # # plt.show()
    # plt.savefig(os.path.join(final_output_dir, 'test_Epoch[{0}]_{1}.png'), dpi=300)
    # plt.close("all")

    # plt.figure(figsize=(18, 6))
    # plot = plt.subplot(121)
    # plt1 = plot.scatter(pca_result_d[:, 0], pca_result_d[:, 1],
    #                     c=tsne_tmol, cmap='rainbow')  # c=pred_choice.cpu().detach().numpy()
    # plt.title('conv_mol_test', x=0.5, y=-0.1)
    # plot = plt.subplot(122)
    # plt2 = plot.scatter(pca_result_d[:, 0], pca_result_d[:, 1],
    #                     c=tsne_tod)  # c=pred_choice.cpu().detach().numpy()
    # plt.title('conv_od_test', x=0.5, y=-0.1)
    # plt.legend(handles=plt2.legend_elements()[0],
    #                       loc='best',
    #                       labels=['order', 'disorder'])  # handles=plot.legend_elements()[0],labels=['order', 'disorder']
    # plt.savefig(os.path.join(final_output_dir, 'pca.png'), dpi=300)

    # plt.figure(figsize=(2, 2))
    # plt.scatter(pca_result_d[:, 0], pca_result_d[:, 1], c=tsne_tmol, cmap='rainbow', s=0.3, lw=0)  # c=pred_choice.cpu().detach().numpy()
    # plt.xlabel('PCA component 1', fontsize=5, va='center')
    # plt.ylabel('PCA component 2', fontsize=5, va='center')
    # plt.title('Global feature PCA', fontsize=5)
    # plt.xticks([-20, -15, -10, -5, 0, 5, 10, 15, 20])
    # plt.yticks([-20, -15, -10, -5, 0, 5, 10, 15, 20])
    # plt.tick_params(axis='both', length=2, pad=2, labelsize=5)
    # # plt.axis('square')
    # plt.tight_layout()
    # plt.savefig(os.path.join(final_output_dir, 'pca_global_feature_mol_valid2.png'), dpi=300)
    # #
    # plt.figure(figsize=(2, 2))
    # ax = plt.scatter(pca_result_d[:, 0], pca_result_d[:, 1], c=tsne_tod, s=0.3, lw=0)  # c=pred_choice.cpu().detach().numpy()
    # plt.xlabel('PCA component 1', fontsize=5, va='center')
    # plt.ylabel('PCA component 2', fontsize=5, va='center')
    # plt.title('Global feature PCA', fontsize=5)
    # plt.xticks([-20, -15, -10, -5, 0, 5, 10, 15, 20])
    # plt.yticks([-20, -15, -10, -5, 0, 5, 10, 15, 20])
    # plt.tick_params(axis='both', length=2, pad=2, labelsize=5)
    # plt.legend(handles=ax.legend_elements()[0], loc=2, labels=['Crystal', 'Liquid'], fontsize=5, markerscale=0.3)
    # # plt.axis('square')
    # plt.tight_layout()
    # plt.savefig(os.path.join(final_output_dir, 'pca_global_feature_od_valid2.png'), dpi=300)

    loss_temp_avg= losses_temp.avg
    #
    sysT = np.array(sysT).reshape(-1)
    targetT = np.array(targetT).reshape(-1)
    predT = np.array(predT).reshape(-1)
    #
    # # id = np.where(predT>800)
    # # predT[id] = targetT[id]
    # #
    deltaT_pred = predT
    deltaT_real = sysT - targetT
    # #
    mae = mean_absolute_error(deltaT_real, deltaT_pred)
    rmse = np.sqrt(mean_squared_error(deltaT_real, deltaT_pred))
    r_2 = r2_score(deltaT_real, deltaT_pred)


    # mae = mean_absolute_error(predT, targetT)
    # rmse = np.sqrt(mean_squared_error(predT, targetT))
    # r_2 = r2_score(predT, targetT)

    # x45 = np.linspace(-50.0, 50.0, 100)
    # y45=x45
    #
    # plt.figure(figsize=(2, 2))
    # plt.plot(x45, y45, c='r', linewidth=0.5)
    # plt.scatter(deltaT_real, deltaT_pred, s=1)
    # # plt.scatter(deltaT_real[8600:15000], deltaT_pred[8600:15000], s=0.7, lw=0, c='blue')
    # # plt.scatter(deltaT_real[7000:8600], deltaT_pred[7000:8600], s=0.7, lw=0, c='orange')
    # # plt.scatter(deltaT_real[15000:], deltaT_pred[15000:], s=0.7, lw=0, c='orange')# c=pred_choice.cpu().detach().numpy()
    # plt.text(-53, 50, s=f"$R^2$={round(r_2, 2)}", fontsize=5)
    # plt.text(-53, 42, 'RMSE={:.2f}'.format(rmse), fontsize=5)
    # plt.text(-53, 34, 'MAE={:.2f}'.format(mae), fontsize=5)
    # plt.xlim([-50.0, 50.0])
    # plt.xticks([-40,-20,0,20,40])
    # plt.ylim([-50.0, 50.0])
    # plt.yticks([-40,-20,0,20,40])
    # plt.tick_params(axis='both', length=2, pad=2, labelsize=5)
    # plt.axis('square')
    # plt.xlabel('True TBMP (K)', fontsize=5, va='center')
    # plt.ylabel('Predicted TBMP (K)', fontsize=5, va='center')
    # plt.tight_layout()
    #
    # plt.savefig(os.path.join(final_output_dir, 'test_Epoch_rmse_735816_2.png'), dpi=300)

    x45 = np.linspace(-100.0, 100.0, 200)
    y45=x45

    plt.figure(figsize=(6, 6))
    plt.scatter(deltaT_real, deltaT_pred, c='r', s=1)  # c=pred_choice.cpu().detach().numpy()
    plt.text(-65, 65, 'R2={:.2f}'.format(r_2))
    plt.text(-65, 60, 'rmse={:.2f}'.format(rmse))
    plt.text(-65, 55, 'mae={:.2f}'.format(mae))
    plt.plot(x45, y45)
    plt.axvline(-40, color='black', linestyle='--')  # shu
    plt.axvline(-30, color='black', linestyle='--')
    plt.axvline(-20, color='black', linestyle='--')
    plt.axvline(-10, color='black', linestyle='--')
    plt.axvline(20, color='black', linestyle='--')
    plt.axvline(30, color='black', linestyle='--')
    plt.axvline(40, color='black', linestyle='--')
    plt.axvline(50, color='black', linestyle='--')
    plt.axhline(-40, color='black', linestyle='--')  # hen
    plt.axhline(-30, color='black', linestyle='--')
    plt.axhline(-20, color='black', linestyle='--')
    plt.axhline(-10, color='black', linestyle='--')
    plt.axhline(20, color='black', linestyle='--')  # hen
    plt.axhline(30, color='black', linestyle='--')
    plt.axhline(40, color='black', linestyle='--')
    plt.axhline(50, color='black', linestyle='--')
    plt.xlim([-70.0, 70.0])
    plt.ylim([-70.0, 70.0])
    plt.xlabel('deltaT_real')
    plt.ylabel('deltaT_predict')

    plt.savefig(os.path.join(final_output_dir, 'test_Epoch_rmse_113810.png'), dpi=720)
    plt.close("all")

    # x45 = np.linspace(0, 60.0, 100)
    # y45=x45
    #
    # plt.figure(figsize=(2, 2))
    # plt.scatter(deltaT_real, deltaT_pred, s=1)  # c=pred_choice.cpu().detach().numpy()
    # plt.text(-1, 61, s=f"$R^2$={round(r_2, 2)}", fontsize=5)
    # plt.text(-1, 56, 'RMSE={:.2f}'.format(rmse), fontsize=5)
    # plt.text(-1, 51, 'MAE={:.2f}'.format(mae), fontsize=5)
    # plt.plot(x45, y45, c='r', linewidth=0.5)
    # plt.xlim([0, 60.0])
    # plt.xticks([0,20,40,60])
    # plt.ylim([0, 60.0])
    # plt.yticks([0,20,40,60])
    # plt.tick_params(axis='both', length=2, pad=2, labelsize=5)
    # plt.axis('square')
    # plt.xlabel('True TBMP (K)', fontsize=5, va='center')
    # plt.ylabel('Predicted TBMP (K)', fontsize=5, va='center')
    # plt.tight_layout()
    #
    # plt.savefig(os.path.join(final_output_dir, 'valid2.png'), dpi=300)
    # plt.close("all")

    # x45 = np.linspace(280.0, 460.0, 180)
    # y45=x45
    # Ttest = targetT[0:70] #s76l70
    # TPtest = predT[0:70]
    # print(TPtest.shape)
    # Tvalid = targetT[70:]
    # TPvalid = predT[70:]
    # print(TPvalid.shape)
    #
    # # maev = mean_absolute_error(TPtest, Ttest)
    # # rmsev = np.sqrt(mean_squared_error(TPtest, Ttest))
    # # r_2v = r2_score(TPtest, Ttest)
    # # print("validation", rmsev, maev, r_2v)
    #
    # # maet = mean_absolute_error(TPvalid, Tvalid)
    # # rmset = np.sqrt(mean_squared_error(TPvalid, Tvalid))
    # # r_2t = r2_score(TPvalid, Tvalid)
    # # print("test", rmset, maet, r_2t)
    # plt.figure(figsize=(2, 2))
    # plt.scatter(Ttest, TPtest , s=1, marker='s', label='Validation')  # c=pred_choice.cpu().detach().numpy()
    # plt.scatter(Tvalid, TPvalid, c='orange', s=1, marker='^', label='Test')  # c=pred_choice.cpu().detach().numpy()
    # plt.text(275, 458, s=f"$R^2$={round(r_2, 2)}", fontsize=5)
    # plt.text(275, 445, 'RMSE={:.2f}'.format(rmse), fontsize=5)
    # plt.text(275, 431, 'MAE={:.2f}'.format(mae), fontsize=5)
    # plt.plot(x45, y45, c='r', linewidth=0.5)
    # plt.xlim([280, 460])
    # plt.xticks(np.arange(280,461,30))
    # plt.ylim([280, 460])
    # plt.yticks(np.arange(280,461,30))
    # plt.tick_params(axis='both', length=2, pad=2, labelsize=5)
    # plt.axis('square')
    # plt.xlabel('True MP (K)', fontsize=5, va='center')
    # plt.ylabel('Predicted MP (K)', fontsize=5, va='center')
    # plt.legend(loc=4, fontsize=5)
    # plt.tight_layout()
    #
    # plt.savefig(os.path.join(final_output_dir, 'rmse_liquid2.png'), dpi=300)
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

def _meter_reduce(meter):
    rank = comm.local_rank
    meter_sum = torch.FloatTensor([meter.sum]).cuda(rank)
    meter_count = torch.FloatTensor([meter.count]).cuda(rank)
    torch.distributed.reduce(meter_sum, 0)
    torch.distributed.reduce(meter_count, 0)
    meter_avg = meter_sum / meter_count

    return meter_avg.item()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
