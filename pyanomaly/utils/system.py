'''
This file is to set up the setting about the system, torch, CUDA, cudnn and so on based on the xxx.yaml
'''
import torch
import logging
import argparse
logger = logging.getLogger(__name__)

def system_setup(args, cfg):
    # cudnn related setting
    torch.backends.cudnn.enable = cfg.SYSTEM.cudnn.enable
    torch.backends.cudnn.benchmark = cfg.SYSTEM.cudnn.benchmark
    torch.backends.cudnn.deterministic = cfg.SYSTEM.cudnn.deterministic
    gpus = cfg.SYSTEM.gpus
    if len(gpus) > 1:
        parallel_flag = True
    elif len(gpus) == 1:
        parallel_flag = False
    else:
        raise Exception('You need to  decide the gpu!')

    if cfg.SYSTEM.distributed.use:
        rank = args.rank * ngpus_per_node + gpu
        logger.info('Need to set the local_rank in args!!!')
        logger.info(f'local_rank:{args.local_rank}')
        torch.distributed.init_process_group(backend="nccl", init_method=args.init_method, world_size=args.world_size, rank=rank)
    
    return parallel_flag


def parse_args():
    parser = argparse.ArgumentParser(description='The CMD commands')

    # General
    parser.add_argument('--cfg_folder', default='debug')
    parser.add_argument('--cfg_name', default='debug.yaml')
    parser.add_argument('--gpus', '-g', default='0', help='Set the using gpus e.g. \'0\', \'0,1\'')
    parser.add_argument('--project_path','-p', default='/export/home/chengyh/PyAnomaly')
    parser.add_argument('--mode', type=str, default='train')
    # parser.add_argument('--inference', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--inference_model', default='')
    parser.add_argument('--local_rank', type=int, default=0) # not use at present
    parser.add_argument('--backend', type=str, default='nccl')
    parser.add_argument('--init_method', type=str, default='tcp://127.0.0.1:23456')
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--verbose', default='debug')
    parser.add_argument('--flow_model_path', default='/export/home/chengyh/Anomaly_DA/lib/networks/liteFlownet/network-sintel.pytorch')
    parser.add_argument('opts', help='change the config from the command-line', default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args