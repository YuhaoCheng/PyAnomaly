"""
@author:  Yuhao Cheng
@contact: yuhao.cheng[at]outlook.com
"""
import torch
from pathlib import Path
from collections import OrderedDict
class EngineAverageMeter(object):
    """
    Computes and store the average the current value
    """
    def __init__(self):
        self.val = 0  # current value 
        self.avg = 0  # avage value
        self.sum = 0  
        self.count = 0 

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def engine_save_model(cfg, cfg_name, model, logger, time_stamp, metric, verbose='None', best=False):
    """Save Models.

    The method is used to save the model into files.

    Args:
        cfg(fvcore.common.config.CfgNode): The config object.
        cfg_name(str): The name of the configuration name.
        model(torch.nn.Module or OrderedDict): The torch.nn.Module or the OrderedDict.
        time_stamp(str): The time stamp when the training process strats.
        metric(float): The performance of this model.
        verbose(str): Comments.
        best(bool): Indicate whether the model is best at present.
    
    Returns:
        output(str): The string of the location(path + name) where stroing the model
    """
    logger.info('=>Save the final model in:{}'.format(cfg.TRAIN.model_output))
    model_name = f'cfg@{cfg_name}#{time_stamp}#{metric:.3f}^{verbose}.pth'
    model_name_best = f'best_{cfg.DATASET.name}_{cfg.MODEL.name}#{metric:.3f}_cfg@{cfg_name}.pth'
    output = Path(cfg.TRAIN.model_output) / cfg.DATASET.name / cfg.MODEL.name 
    output.mkdir(parents=True, exist_ok=True)
    
    output = output / model_name
    output_best = Path(cfg.TRAIN.model_output) / model_name_best
    if type(model) == type(dict()) or type(model) == type(OrderedDict()):
        temp = {k:v.state_dict() for k, v in model.items()}
        torch.save(temp, output)
        if best:
            torch.save(temp, output_best)
    else:
        torch.save(model.state_dict(), output)
        if best:
            torch.save(model.state_dict(), output_best)

    logger.info(f'\033[1;31m =>Saved Model name:{model_name} \033[0m')
    logger.info(f'\033[1;32m =>Saved Best Model name:{model_name_best} \033[0m')

    return str(output)


def engine_save_checkpoint(cfg, cfg_name, model, epoch, loss, optimizer, logger, time_stamp, metric, flag='inter', verbose='None', best=False):
    """Save the checkpoint.
    The method is used to save the checkpoint of training including the models and optimizer, in order to resume the training process\
    
    Args:
        cfg(fvcore.common.config.CfgNode): The config object.

        flag(str): if the cheeckpoint is final, the value of it is 'final'. else, the value of it is 'inter'

    """
    output = Path(cfg.TRAIN.checkpoint_output) / cfg.DATASET.name / cfg.MODEL.name /('cfg@' + cfg_name) / time_stamp
    output.mkdir(parents=True, exist_ok=True)
    logger.info(f'=>Save the checkpoint in:{output}')
    
    # Save in a dict
    checkpoint = OrderedDict() # make the type of the checkpoint is OrderedDict 
    checkpoint['epoch'] = epoch
    checkpoint['loss'] = loss
    # Models
    if type(model) == type(dict()) or type(model) == type(OrderedDict()):
        temp = {k:v.state_dict() for k, v in model.items()}
        checkpoint['model_state_dict'] = temp
    else:
        checkpoint['model_state_dict'] = model.state_dict()
    
    # Optimizer
    if type(optimizer) == type(dict()) or type(optimizer) == type(OrderedDict()):
        temp = {k:v.state_dict() for k, v in optimizer.items()}
        checkpoint['optimizer_state_dict'] = temp
    else:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    # Save
    file_name = f'{flag}_epoch{epoch}#{metric:.3f}^{verbose}.pth.tar'
    file_name_best = f'best_ckpt_{cfg.DATASET.name}_{cfg.MODEL.name}#{metric:.3f}_cfg@{cfg_name}.pth'
    output = output / file_name
    output_best = Path(cfg.TRAIN.checkpoint_output) / file_name_best
    torch.save(checkpoint, output)
    logger.info(f'\033[1;34m =>Save checkpoint:{file_name} \033[0m')
    if best:
        torch.save(checkpoint, output_best)
        logger.info(f'\033[1;32m =>Save Best checkpoint:{file_name} \033[0m')
