import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter

from sklearn.svm import LinearSVC
import numpy as np
from torchvision import transforms
from datasets import data_transforms
from pointnet2_ops import pointnet2_utils
from torchstat import stat

train_transforms = transforms.Compose(
    [
        # data_transforms.PointcloudScale(),
        # data_transforms.PointcloudRotate(),
        # data_transforms.PointcloudRotatePerturbation(),
        # data_transforms.PointcloudTranslate(),
        # data_transforms.PointcloudJitter(),
        # data_transforms.PointcloudRandomInputDropout(),
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)


class Acc_Metric:
    def __init__(self, acc=0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict


def evaluate_svm(train_features, train_labels, test_features, test_labels):
    clf = LinearSVC()
    clf.fit(train_features, train_labels)
    pred = clf.predict(test_features)
    return np.sum(test_labels == pred) * 1. / pred.shape[0]


def _unpack_pretrain_output(model_out):
    if isinstance(model_out, tuple):
        loss = model_out[0]
        diag = model_out[1] if len(model_out) > 1 else None
    else:
        loss = model_out
        diag = None

    if torch.is_tensor(loss) and loss.dim() > 0:
        loss = loss.mean()
    if torch.is_tensor(diag) and diag.dim() > 1:
        diag = diag.mean(dim=0)

    return loss, diag


def _fmt_diag(value):
    if value != value:
        return 'nan'
    return f'{value:.4f}'


def run_net(args, config, train_writer=None, val_writer=None, wandb_run=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader), = builder.dataset_builder(args, config.dataset.train), \
        builder.dataset_builder(args, config.dataset.val)
    (_, extra_train_dataloader) = builder.dataset_builder(
        args, config.dataset.extra_train) if config.dataset.get('extra_train') else (None, None)
    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # from IPython import embed; embed()

    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(
            base_model, args, logger=logger)
        best_metrics = Acc_Metric(best_metric)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger=logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[
                                                         args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)

    # trainval
    # training
    base_model.zero_grad()
    wandb_log_every = max(1, int(getattr(args, 'wandb_log_every', 10)))
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['Loss'])

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx

            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train.others.npoints
            dataset_name = config.dataset.train._base_.NAME
            if dataset_name == 'ShapeNet' or dataset_name == 'UnlabeledHybrid':
                points = data.cuda()
            elif dataset_name == 'ModelNet':
                points = data[0].cuda()
                points = misc.fps(points, npoints)
            else:
                raise NotImplementedError(
                    f'Train phase do not support {dataset_name}')

            assert points.size(1) == npoints
            points = train_transforms(points)
            loss, pretrain_diag = _unpack_pretrain_output(base_model(points))
            try:
                loss.backward()
            except:
                loss = loss.mean()
                loss.backward()

            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                if torch.is_tensor(pretrain_diag):
                    pretrain_diag = dist_utils.reduce_tensor(pretrain_diag, args)
                losses.update([loss.item()*1000])
            else:
                losses.update([loss.item()*1000])

            loss_main_val = float('nan')
            cos_tgt_val = float('nan')
            cos_prev_val = float('nan')
            gap_val = float('nan')
            copy_win_val = float('nan')
            recon_cd_l1_val = float('nan')
            recon_cd_l2_val = float('nan')
            if torch.is_tensor(pretrain_diag) and pretrain_diag.numel() >= 5:
                loss_main_val = float(pretrain_diag[0].item())
                cos_tgt_val = float(pretrain_diag[1].item())
                cos_prev_val = float(pretrain_diag[2].item())
                gap_val = float(pretrain_diag[3].item())
                copy_win_val = float(pretrain_diag[4].item())
                if pretrain_diag.numel() >= 7:
                    recon_cd_l1_val = float(pretrain_diag[5].item())
                    recon_cd_l2_val = float(pretrain_diag[6].item())

            if args.distributed:
                torch.cuda.synchronize()

            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                if loss_main_val == loss_main_val:
                    train_writer.add_scalar('Loss/Batch/LossMain', loss_main_val, n_itr)
                if cos_tgt_val == cos_tgt_val:
                    train_writer.add_scalar('Diag/Batch/CosTgt', cos_tgt_val, n_itr)
                if cos_prev_val == cos_prev_val:
                    train_writer.add_scalar('Diag/Batch/CosPrev', cos_prev_val, n_itr)
                if gap_val == gap_val:
                    train_writer.add_scalar('Diag/Batch/Gap', gap_val, n_itr)
                if copy_win_val == copy_win_val:
                    train_writer.add_scalar('Diag/Batch/CopyWin', copy_win_val, n_itr)
                if recon_cd_l1_val == recon_cd_l1_val:
                    train_writer.add_scalar('Diag/Batch/ReconCDL1', recon_cd_l1_val, n_itr)
                if recon_cd_l2_val == recon_cd_l2_val:
                    train_writer.add_scalar('Diag/Batch/ReconCDL2', recon_cd_l2_val, n_itr)
                train_writer.add_scalar(
                    'Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

            if wandb_run is not None and (n_itr % wandb_log_every == 0):
                wb = {
                    'train/loss': float(loss.item()),
                    'train/loss_x1k': float(loss.item() * 1000.0),
                    'train/lr': float(optimizer.param_groups[0]['lr']),
                    'train/step': int(n_itr),
                    'train/epoch': float(epoch) + float(idx + 1) / float(max(1, n_batches)),
                }
                if loss_main_val == loss_main_val:
                    wb['diag/loss_main'] = loss_main_val
                if cos_tgt_val == cos_tgt_val:
                    wb['diag/cos_tgt'] = cos_tgt_val
                if cos_prev_val == cos_prev_val:
                    wb['diag/cos_prev'] = cos_prev_val
                if gap_val == gap_val:
                    wb['diag/gap'] = gap_val
                if copy_win_val == copy_win_val:
                    wb['diag/copy_win'] = copy_win_val
                if recon_cd_l1_val == recon_cd_l1_val:
                    wb['diag/recon_cd_l1'] = recon_cd_l1_val
                if recon_cd_l2_val == recon_cd_l2_val:
                    wb['diag/recon_cd_l2'] = recon_cd_l2_val
                wandb_run.log(wb, step=int(n_itr))

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f '
                          'diag(loss_main=%s cos_tgt=%s cos_prev=%s gap=%s copy_win=%s recon_cd_l1=%s recon_cd_l2=%s)' %
                          (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                           ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr'],
                           _fmt_diag(loss_main_val), _fmt_diag(cos_tgt_val), _fmt_diag(cos_prev_val),
                           _fmt_diag(gap_val), _fmt_diag(copy_win_val),
                           _fmt_diag(recon_cd_l1_val), _fmt_diag(recon_cd_l2_val)), logger=logger)
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss_1', losses.avg(0), epoch)
        if wandb_run is not None:
            wandb_run.log(
                {
                    'epoch/loss_x1k': float(losses.avg(0)),
                    'epoch/index': int(epoch),
                },
                step=int((epoch + 1) * max(1, n_batches)),
            )
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
                  (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],
                   optimizer.param_groups[0]['lr']), logger=logger)

        # if epoch % args.val_freq == 0 and epoch != 0:
        #     # Validate the current model
        #     metrics = validate(base_model, extra_train_dataloader, test_dataloader, epoch, val_writer, args, config, logger=logger)
        #
        #     # Save ckeckpoints
        #     if metrics.better_than(best_metrics):
        #         best_metrics = metrics
        #         builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
        builder.save_checkpoint(base_model, optimizer, epoch,
                                metrics, best_metrics, 'ckpt-last', args, logger=logger)
        if epoch % 25 == 0 and epoch >= 250:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args,
                                    logger=logger)
        # if (config.max_epoch - epoch) < 10:
        #     builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()


def validate(base_model, extra_train_dataloader, test_dataloader, epoch, val_writer, args, config, logger=None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger=logger)
    base_model.eval()  # set model to eval mode

    test_features = []
    test_label = []

    train_features = []
    train_label = []
    npoints = config.dataset.train.others.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(extra_train_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)

            assert points.size(1) == npoints
            feature = base_model(points, noaug=True)
            target = label.view(-1)

            train_features.append(feature.detach())
            train_label.append(target.detach())

        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)
            assert points.size(1) == npoints
            feature = base_model(points, noaug=True)
            target = label.view(-1)

            test_features.append(feature.detach())
            test_label.append(target.detach())

        train_features = torch.cat(train_features, dim=0)
        train_label = torch.cat(train_label, dim=0)
        test_features = torch.cat(test_features, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            train_features = dist_utils.gather_tensor(train_features, args)
            train_label = dist_utils.gather_tensor(train_label, args)
            test_features = dist_utils.gather_tensor(test_features, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        svm_acc = evaluate_svm(train_features.data.cpu().numpy(), train_label.data.cpu(
        ).numpy(), test_features.data.cpu().numpy(), test_label.data.cpu().numpy())

        print_log('[Validation] EPOCH: %d  acc = %.4f' %
                  (epoch, svm_acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', svm_acc, epoch)

    return Acc_Metric(svm_acc)


def test_net():
    pass
