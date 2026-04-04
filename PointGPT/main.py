from tools import pretrain_run_net as pretrain
from tools import finetune_run_net as finetune
from tools import test_run_net as test_net
from utils import parser, dist_utils, misc
from utils.logger import *
from utils.config import *
import time
import os
import torch
from tensorboardX import SummaryWriter
from torchstat import stat


def _to_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, 'items'):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if hasattr(obj, '__dict__'):
        return {str(k): _to_serializable(v) for k, v in vars(obj).items()}
    return str(obj)


def _init_wandb(args, config, logger=None):
    if int(args.use_wandb) != 1 or args.test:
        return None
    if args.distributed:
        rank, _ = dist_utils.get_dist_info()
        if rank != 0:
            return None
    try:
        import wandb  # type: ignore
    except Exception as e:
        print_log(f'[wandb] disabled: import failed ({e})', logger=logger)
        return None

    mode = str(args.wandb_mode).strip()
    if mode == 'disabled':
        print_log('[wandb] disabled by mode=disabled', logger=logger)
        return None

    tags = [t.strip() for t in str(args.wandb_tags).split(',') if t.strip()]
    run_name = str(args.wandb_run_name).strip() or str(args.exp_name)
    group = str(args.wandb_group).strip() or None
    entity = str(args.wandb_entity).strip() or None

    init_kwargs = dict(
        project=str(args.wandb_project),
        entity=entity,
        name=run_name,
        group=group,
        tags=tags,
        mode=mode,
        config={
            'args': _to_serializable(vars(args)),
            'config': _to_serializable(config),
        },
    )
    if str(args.wandb_dir).strip():
        init_kwargs['dir'] = str(args.wandb_dir).strip()

    try:
        run = wandb.init(**init_kwargs)
        print_log(
            f'[wandb] enabled project={args.wandb_project} run={run_name} '
            f'group={group if group else "-"} mode={mode}',
            logger=logger,
        )
        return run
    except Exception as e:
        print_log(f'[wandb] disabled: init failed ({e})', logger=logger)
        return None


def main():
    # args
    args = parser.get_args()
    # CUDA
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        args.distributed = False
    else:
        args.distributed = True
        dist_utils.init_dist(args.launcher)
        # re-set gpu_ids with distributed training mode
        _, world_size = dist_utils.get_dist_info()
        args.world_size = world_size
    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)
    # define the tensorboard writer
    if not args.test:
        if args.local_rank == 0:
            train_writer = SummaryWriter(
                os.path.join(args.tfboard_path, 'train'))
            val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'test'))
        else:
            train_writer = None
            val_writer = None
    # config
    config = get_config(args, logger=logger)
    # batch size
    if args.distributed:
        assert config.total_bs % world_size == 0
        config.dataset.train.others.bs = config.total_bs // world_size
        if config.dataset.get('extra_train'):
            config.dataset.extra_train.others.bs = config.total_bs // world_size * 2
        config.dataset.val.others.bs = config.total_bs // world_size * 2
        if config.dataset.get('test'):
            config.dataset.test.others.bs = config.total_bs // world_size
    else:
        config.dataset.train.others.bs = config.total_bs
        if config.dataset.get('extra_train'):
            config.dataset.extra_train.others.bs = config.total_bs * 2
        config.dataset.val.others.bs = config.total_bs * 2
        if config.dataset.get('test'):
            config.dataset.test.others.bs = config.total_bs
    # log
    log_args_to_file(args, 'args', logger=logger)
    log_config_to_file(config, 'config', logger=logger)
    # exit()
    logger.info(f'Distributed training: {args.distributed}')
    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        # seed + rank, for augmentation
        misc.set_random_seed(args.seed + args.local_rank,
                             deterministic=args.deterministic)
    if args.distributed:
        assert args.local_rank == torch.distributed.get_rank()

    wandb_run = _init_wandb(args, config, logger=logger)

    if args.shot != -1:
        config.dataset.train.others.shot = args.shot
        config.dataset.train.others.way = args.way
        config.dataset.train.others.fold = args.fold
        config.dataset.val.others.shot = args.shot
        config.dataset.val.others.way = args.way
        config.dataset.val.others.fold = args.fold

    # run
    try:
        if args.test:
            test_net(args, config)
        else:
            if args.finetune_model or args.scratch_model:
                finetune(args, config, train_writer, val_writer, wandb_run=wandb_run)
            else:
                pretrain(args, config, train_writer, val_writer, wandb_run=wandb_run)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == '__main__':
    main()
