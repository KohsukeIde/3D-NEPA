import os
import argparse
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', 
        type = str, 
        help = 'yaml config file')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default='none',
        help='job launcher')     
    parser.add_argument('--local_rank', type=int, default=int(os.environ.get('LOCAL_RANK', '0')))
    parser.add_argument('--num_workers', type=int, default=8)
    # seed 
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')      
    # bn
    parser.add_argument(
        '--sync_bn', 
        action='store_true', 
        default=False, 
        help='whether to use sync bn')
    # some args
    parser.add_argument('--exp_name', type = str, default='default', help = 'experiment name')
    parser.add_argument('--loss', type=str, default='cd1', help='loss name')
    parser.add_argument('--start_ckpts', type = str, default=None, help = 'reload used ckpt path')
    parser.add_argument('--ckpts', type = str, default=None, help = 'test used ckpt path')
    parser.add_argument('--val_freq', type = int, default=1, help = 'test freq')
    parser.add_argument(
        '--vote',
        action='store_true',
        default=False,
        help = 'vote acc')
    parser.add_argument(
        '--resume', 
        action='store_true', 
        default=False, 
        help = 'autoresume training (interrupted by accident)')
    parser.add_argument(
        '--test', 
        action='store_true', 
        default=False, 
        help = 'test mode for certain ckpt')
    parser.add_argument(
        '--finetune_model', 
        action='store_true', 
        default=False, 
        help = 'finetune modelnet with pretrained weight')
    parser.add_argument(
        '--scratch_model', 
        action='store_true', 
        default=False, 
        help = 'training modelnet from scratch')
    parser.add_argument(
        '--mode', 
        choices=['easy', 'median', 'hard', None],
        default=None,
        help = 'difficulty mode for shapenet')        
    parser.add_argument(
        '--way', type=int, default=-1)
    parser.add_argument(
        '--shot', type=int, default=-1)
    parser.add_argument(
        '--fold', type=int, default=-1)
    parser.add_argument(
        '--use_wandb',
        type=int,
        default=int(os.environ.get('USE_WANDB', '0')),
        choices=[0, 1],
        help='enable Weights & Biases logging')
    parser.add_argument(
        '--wandb_project',
        type=str,
        default=os.environ.get('WANDB_PROJECT', 'pointgpt-pretrain'),
        help='W&B project name')
    parser.add_argument(
        '--wandb_entity',
        type=str,
        default=os.environ.get('WANDB_ENTITY', ''),
        help='W&B entity/team')
    parser.add_argument(
        '--wandb_run_name',
        type=str,
        default=os.environ.get('WANDB_RUN_NAME', ''),
        help='W&B run name override')
    parser.add_argument(
        '--wandb_group',
        type=str,
        default=os.environ.get('WANDB_GROUP', ''),
        help='W&B group name')
    parser.add_argument(
        '--wandb_tags',
        type=str,
        default=os.environ.get('WANDB_TAGS', ''),
        help='comma-separated W&B tags')
    parser.add_argument(
        '--wandb_mode',
        type=str,
        default=os.environ.get('WANDB_MODE', 'online'),
        choices=['online', 'offline', 'disabled'],
        help='W&B mode')
    parser.add_argument(
        '--wandb_log_every',
        type=int,
        default=int(os.environ.get('WANDB_LOG_EVERY', '10')),
        help='W&B batch logging interval')
    parser.add_argument(
        '--wandb_dir',
        type=str,
        default=os.environ.get('WANDB_DIR', ''),
        help='optional W&B local directory')
    parser.add_argument(
        '--ft_recon_weight',
        type=float,
        default=float(os.environ.get('FT_RECON_WEIGHT', '3.0')),
        help='auxiliary reconstruction loss weight during fine-tuning; set 0 for cls-only FT')
    parser.add_argument(
        '--save_last_every_epoch',
        type=int,
        default=int(os.environ.get('SAVE_LAST_EVERY_EPOCH', '1')),
        choices=[0, 1],
        help='save ckpt-last at every epoch during fine-tuning; set 0 to save only once at the end')
    parser.add_argument(
        '--test_vote_times',
        type=int,
        default=int(os.environ.get('TEST_VOTE_TIMES', '299')),
        help='number of repeated test_vote evaluations in --test mode; set 0 to skip test-time voting')
    
    args = parser.parse_args()

    if args.test and args.resume:
        raise ValueError(
            '--test and --resume cannot be both activate')

    if args.resume and args.start_ckpts is not None:
        raise ValueError(
            '--resume and --start_ckpts cannot be both activate')

    if args.test and args.ckpts is None:
        raise ValueError(
            'ckpts shouldnt be None while test mode')

    if args.finetune_model and args.ckpts is None:
        print(
            'training from scratch')

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.test:
        args.exp_name = 'test_' + args.exp_name
    if args.mode is not None:
        args.exp_name = args.exp_name + '_' +args.mode
    args.experiment_path = os.path.join('./experiments', Path(args.config).stem, Path(args.config).parent.stem, args.exp_name)
    args.tfboard_path = os.path.join('./experiments', Path(args.config).stem, Path(args.config).parent.stem,'TFBoard' ,args.exp_name)
    args.log_name = Path(args.config).stem
    create_experiment_dir(args)
    return args

def create_experiment_dir(args):
    exp_created = not os.path.exists(args.experiment_path)
    os.makedirs(args.experiment_path, exist_ok=True)
    if exp_created:
        print('Create experiment path successfully at %s' % args.experiment_path)
    tf_created = not os.path.exists(args.tfboard_path)
    os.makedirs(args.tfboard_path, exist_ok=True)
    if tf_created:
        print('Create TFBoard path successfully at %s' % args.tfboard_path)
