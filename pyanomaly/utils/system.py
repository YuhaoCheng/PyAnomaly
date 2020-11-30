'''
This file is to set up the setting about the system, torch, CUDA, cudnn and so on based on the xxx.yaml
'''
import torch
import logging
logger = logging.getLogger(__name__)
# def system_setup(args, ngpus_per_node, gpu,cfg, logger):
# def system_setup(args, cfg, logger):
def system_setup(args, cfg):
    # cudnn related setting
    torch.backends.cudnn.enable = cfg.SYSTEM.cudnn.enable
    torch.backends.cudnn.benchmark = cfg.SYSTEM.cudnn.benchmark
    torch.backends.cudnn.deterministic = cfg.SYSTEM.cudnn.deterministic
    if cfg.SYSTEM.distributed.use:
        rank = args.rank * ngpus_per_node + gpu
        logger.info('Need to set the local_rank in args!!!')
        logger.info(f'local_rank:{args.local_rank}')
        torch.distributed.init_process_group(backend="nccl", init_method=args.init_method, world_size=args.world_size, rank=rank)