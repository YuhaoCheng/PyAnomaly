import os
import logging 
import time
import torch
from pathlib import Path
from collections import OrderedDict
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from lib.config.config import update_config

from colorama import init,Fore
init(autoreset=True)

def create_logger(root_path, cfg, cfg_name, phase='trian', verbose='None'):
    '''
    Create the root logger. 
    The rest of log file is using the same time as this time
    Args:
        root_path: Path object, the root path of the project
        cfg: the config class of the whole process
        cfg_name: the name of the config file(yaml file)
        phase: the flag indicate the stage, trian, val or test
        verbose: some note
    Returns:
        logger: the logger instance
        final_output_dir: the dir of final output to store the results, such as the accuracy, the images or some thing
        tensorboard_log_dir: 
        cfg_name 
        time_str
    '''
    root_output_dir = root_path / cfg.LOG.log_output_dir
    
    # set up the logger
    if not root_output_dir.exists():
        root_output_dir.mkdir(parents=True)
        print('=> Creating the Log Root{}'.format(root_output_dir))
    
    dataset = cfg.DATASET.name
    model = cfg.MODEL.name
    cfg_name = os.path.basename(cfg_name)[0:-5] # in order to cfg name includes the '.', e.g. cascade_det0.4_L.yaml

    final_output_dir = root_output_dir / dataset / model / cfg_name

    final_output_dir.mkdir(parents=True, exist_ok=True)
    print('=> Creating the Log folder: {}'.format(final_output_dir))

    time_str = time.strftime('%Y-%m-%d-%H-%M') # 2019-08-07-10-34
    log_file =f'cfg@{cfg_name}_{phase}_{verbose}_{time_str}.log'

    # final log file path
    final_log_file = final_output_dir / log_file
    print(f'=> log file is:{final_log_file}')

    vis_dir = root_output_dir / 'vis'
    vis_dir.mkdir(parents=False, exist_ok=True)
    print(f'=> the vis dir is {str(vis_dir)}')
    
    # result_path = root_output_dir / 'results'
    # if not result_path.exists:
    #     print(f'=> make the results folder:{str(result_path)}')

    result_path = root_output_dir / 'results'
    if not result_path.exists:
        print(f'=> make the results folder:{str(result_path)}')
    
    # set up the basic of the logger
    logger = logging.getLogger()
    fmt = '%(asctime)-15s:%(message)s'
    datefmt = '%Y-%m-%d-%H:%M'
    formatter = logging.Formatter(fmt=fmt,datefmt=datefmt)
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    file_handler = logging.FileHandler(final_log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)
    
    # set up the path of the tensorboard
    tensorboard_log_dir = Path(cfg.LOG.tb_output_dir) / dataset / model / f'cfg@{cfg_name}' / phase / verbose / time_str
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
    print(Fore.RED + '=> Create the tensorboard folder:{}'.format(tensorboard_log_dir))

    return logger, str(final_output_dir), str(tensorboard_log_dir), cfg_name, time_str, str(final_log_file)

def get_tensorboard(tensorboard_log_dir, time_stamp, model_name, final_log_file_name):
    '''
    Get the tensorboard writer of
    Args:
        tensorboard_log_dir: the root of the tensorboard
        time_stamp: the time when the training start
        model_name: the model type 
    '''
    print('=> Create the writer of tensorboard')

    if not os.path.exists(tensorboard_log_dir):
        raise Exception('!!!!!!!!Not create the logger or something wrong in creating logger!!!!!!!')
    writer_dict = {'writer': SummaryWriter(log_dir=tensorboard_log_dir),
                    f'global_steps_{model_name}': 0,
                    'time_stamp': time_stamp
                    }
    writer_dict['writer'].add_text('log_file_name', final_log_file_name, 0)
    return writer_dict

def save_model(cfg, cfg_name, model, logger, time_stamp,metric,verbose='None'):
    '''
    Save the final model of training 
    '''
    logger.info('=>Save the final model in:{}'.format(cfg.TRAIN.model_output))
    # time_str = time.strftime('%Y-%m-%d-%H-%M')
    # cfg_name = os.path.basename(cfg_name).split('.')[0]
    # cfg_name = os.path.basename(cfg_name)[0:-5] # in order to cfg name includes the '.', e.g. cascade_det0.4_L.yaml
    model_name = cfg.DATASET.name + '_' + cfg.MODEL.name + '_cfg' + cfg_name + '#' + time_stamp + '#' + '{:.3f}'.format(metric) + '^' + verbose + '.pth'

    output = Path(cfg.TRAIN.model_output) / cfg.DATASET.name 
    output.mkdir(parents=True, exist_ok=True)
    
    output = output / model_name
    torch.save(model.state_dict(), output)

    logger.info('=>NOT IN THE ENGINE!!!Saved Model name:{}'.format(model_name))

    return str(output)
    

def save_checkpoint(cfg, cfg_name, model, epoch, loss, optimizer, logger, time_stamp, metric, flag='inter', verbose='None'):
    '''
    Save the checkpoint of training, in order to resume the training process
    Args:
        flag: if the cheeckpoint is final, the value of it is 'final'. else, the value of it is 'inter'
    '''
    # cfg_name = os.path.basename(cfg_name).split('.')[0]
    # cfg_name = os.path.basename(cfg_name)[0:-5] # in order to cfg name includes the '.', e.g. cascade_det0.4_L.yaml
    output = Path(cfg.TRAIN.checkpoint_output) / cfg.DATASET.name / ('cfg' + cfg_name) / cfg.MODEL.name / time_stamp
    output.mkdir(parents=True, exist_ok=True)
    logger.info('=>Save the checkpoint in:{}'.format(output))
    
    # Save in a dict
    # checkpoint = dict()
    checkpoint = OrderedDict() # make the type of the checkpoint is OrderedDict 
    checkpoint['epoch'] = epoch
    checkpoint['loss'] = loss
    checkpoint['model_state_dict'] = model.state_dict()
    checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    # Save
    # time_str = time.strftime('%Y-%m-%d-%H-%M')
    file_name = '{flag}_epoch{epoch}#{metric:.2f}^{verbose}.pth.tar'.format(flag=flag, epoch=epoch, metric=metric, verbose=verbose)
    
    output = output / file_name
    torch.save(checkpoint, output)
    logger.info('=>NOT IN THE ENGINE!!!Save checkpoint:{}'.format(file_name))


if __name__ == '__main__':
    # lg, final_output_dir, tensorboard_log_dir, cfg_name = create_logger('test', 'test')
    # lg.info('123')
    # w_d = get_tensorboard('./test')
    cfg = update_config('/export/home/chengyh/aaai2020/crowdaction/experiments/0819/mca_baseline.yaml')
    label = [[9,10,18,0]]
    one_hot_label = one_hot(label, 1, cfg.DATASET.number_of_class)
    print(one_hot_label)
    print(one_hot_label.shape)
    





