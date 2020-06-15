import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='The CMD commands')

    # General
    parser.add_argument('--cfg_folder', default='debug')
    parser.add_argument('--cfg_name', default='debug.yaml')
    parser.add_argument('--project_path','-p', default='/export/home/chengyh/Anomaly_DA')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--inference', action='store_true')
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