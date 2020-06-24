import torch
from colorama import init, Fore, Back
init(autoreset=True)
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

def engine_save_model(cfg, cfg_name, model, logger, time_stamp,metric,verbose='None',best=False):
    '''
    Save the final model of training 
    '''
    logger.info('=>Save the final model in:{}'.format(cfg.TRAIN.model_output))
    model_name = f'cfg@{cfg_name}#{time_stamp}#{metric:.3f}^{verbose}.pth'
    model_name_best = f'best_{cfg.DATASET.name}_{cfg.MODEL.name}#{metric:.3f}_cfg@{cfg_name}.pth'
    output = Path(cfg.TRAIN.model_output) / cfg.DATASET.name / cfg.MODEL.name 
    output.mkdir(parents=True, exist_ok=True)
    
    output = output / model_name
    output_best = Path(cfg.TRAIN.model_output) / model_name_best
    if type(model) == type(dict()):
        temp = {k:v.state_dict() for k, v in model.items()}
        torch.save(temp, output)
        if best:
            torch.save(temp, output_best)
    else:
        torch.save(model.state_dict(), output)
        if best:
            torch.save(model.state_dict(), output_best)

    logger.info(Fore.RED + f'=>Saved Model name:{model_name}')
    logger.info(Fore.GREEN + f'=>Saved Best Model name:{model_name_best}')

    return str(output)


def engine_save_checkpoint(cfg, cfg_name, model, epoch, loss, optimizer, logger, time_stamp, metric, flag='inter', verbose='None', best=False):
    '''
    Save the checkpoint of training, in order to resume the training process
    Args:
        flag: if the cheeckpoint is final, the value of it is 'final'. else, the value of it is 'inter'
    '''
    output = Path(cfg.TRAIN.checkpoint_output) / cfg.DATASET.name / cfg.MODEL.name /('cfg@' + cfg_name) / time_stamp
    output.mkdir(parents=True, exist_ok=True)
    logger.info(f'=>Save the checkpoint in:{output}')
    
    # Save in a dict
    checkpoint = OrderedDict() # make the type of the checkpoint is OrderedDict 
    checkpoint['epoch'] = epoch
    checkpoint['loss'] = loss
    if type(model) == type(dict()):
        temp = {k:v.state_dict() for k, v in model.items()}
        checkpoint['model_state_dict'] = temp
    else:
        checkpoint['model_state_dict'] = model.state_dict()
    if type(optimizer) == type(dict()):
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
    logger.info(Fore.BLUE + f'=>Save checkpoint:{file_name}')
    if best:
        torch.save(checkpoint, output_best)
        logger.info(Fore.GREEN + f'=>Save Best checkpoint:{file_name}')
