"""
@author:  Yuhao Cheng
@contact: yuhao.cheng[at]outlook.com
"""
import torch
from pathlib import Path
from collections import OrderedDict
import logging
logger = logging.getLogger(__name__)
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

# def engine_save_model(cfg, cfg_name, model, logger, time_stamp, metric, verbose='None', best=False):
#     """Save Models.

#     The method is used to save the model into files.

#     Args:
#         cfg(fvcore.common.config.CfgNode): The config object.
#         cfg_name(str): The name of the configuration name.
#         model(torch.nn.Module or OrderedDict): The torch.nn.Module or the OrderedDict.
#         time_stamp(str): The time stamp when the training process strats.
#         metric(float): The performance of this model.
#         verbose(str): Comments.
#         best(bool): Indicate whether the model is best at present.
    
#     Returns:
#         output(str): The string of the location(path + name) where stroing the model
#     """
#     logger.info('=>Save the final model in:{}'.format(cfg.TRAIN.model_output))
#     model_name = f'cfg@{cfg_name}#{time_stamp}#{metric:.3f}^{verbose}.pth'
#     model_name_best = f'best_{cfg.DATASET.name}_{cfg.MODEL.name}#{metric:.3f}_cfg@{cfg_name}.pth'
#     output = Path(cfg.TRAIN.model_output) / cfg.DATASET.name / cfg.MODEL.name 
#     output.mkdir(parents=True, exist_ok=True)
    
#     output = output / model_name
#     output_best = Path(cfg.TRAIN.model_output) / model_name_best
#     if type(model) == type(dict()) or type(model) == type(OrderedDict()):
#         temp = {k:v.state_dict() for k, v in model.items()}
#         torch.save(temp, output)
#         if best:
#             torch.save(temp, output_best)
#     else:
#         torch.save(model.state_dict(), output)
#         if best:
#             torch.save(model.state_dict(), output_best)

#     logger.info(f'\033[1;31m =>Saved Model name:{model_name} \033[0m')
#     logger.info(f'\033[1;32m =>Saved Best Model name:{model_name_best} \033[0m')

#     return str(output)


# def engine_save_checkpoint(cfg, cfg_name, model, epoch, loss, optimizer, logger, time_stamp, metric, flag='inter', verbose='None', best=False):
#     """Save the checkpoint.
#     The method is used to save the checkpoint of training including the models and optimizer, in order to resume the training process\
    
#     Args:
#         cfg(fvcore.common.config.CfgNode): The config object.

#         flag(str): if the cheeckpoint is final, the value of it is 'final'. else, the value of it is 'inter'

#     """
#     output = Path(cfg.TRAIN.checkpoint_output) / cfg.DATASET.name / cfg.MODEL.name /('cfg@' + cfg_name) / time_stamp
#     output.mkdir(parents=True, exist_ok=True)
#     logger.info(f'=>Save the checkpoint in:{output}')
    
#     # Save in a dict
#     checkpoint = OrderedDict() # make the type of the checkpoint is OrderedDict 
#     checkpoint['epoch'] = epoch
#     checkpoint['loss'] = loss
#     # Models
#     if type(model) == type(dict()) or type(model) == type(OrderedDict()):
#         temp = {k:v.state_dict() for k, v in model.items()}
#         checkpoint['model_state_dict'] = temp
#     else:
#         checkpoint['model_state_dict'] = model.state_dict()
    
#     # Optimizer
#     if type(optimizer) == type(dict()) or type(optimizer) == type(OrderedDict()):
#         temp = {k:v.state_dict() for k, v in optimizer.items()}
#         checkpoint['optimizer_state_dict'] = temp
#     else:
#         checkpoint['optimizer_state_dict'] = optimizer.state_dict()

#     # Save
#     file_name = f'{flag}_epoch{epoch}#{metric:.3f}^{verbose}.pth.tar'
#     file_name_best = f'best_ckpt_{cfg.DATASET.name}_{cfg.MODEL.name}#{metric:.3f}_cfg@{cfg_name}.pth'
#     output = output / file_name
#     output_best = Path(cfg.TRAIN.checkpoint_output) / file_name_best
#     torch.save(checkpoint, output)
#     logger.info(f'\033[1;34m =>Save checkpoint:{file_name} \033[0m')
#     if best:
#         torch.save(checkpoint, output_best)
#         logger.info(f'\033[1;32m =>Save Best checkpoint:{file_name} \033[0m')
def engine_save(cfg, cfg_name, saved_stuff, step, time_stamp, metric, flag='inter', verbose='None', best=False, save_model=False):
    """Save the checkpoint file.
    Save the checkpoint of training, in order to resume the training process. 
    The saving ckpt path is: OUTPUT_DIR / DATASET.DATASET / MODEL.NAME / cfg@xxx / time_stamp / xxx.pth
    The saving model path is: OUTPUT_DIR / DATASET.DATASET / MODEL.NAME / xxx.pth
    The saving structure is:{
        'epoch': xxx
        'loss': xxx
        'model_state_dict': {
            $model_name_in_dict: xxxx
        }
        'optimizer_state_dict': xxx
    }
    Args:
        cfg: The configuration object
        cfg_name: the name of the configuration file
        saved_stuff(dict): The saving things incules the model, epoch, loss, optimizer. At least, it contains the model. For example:
            {
                'step': 0,
                'loss':0,
                $model_name: xxxx,
                $optimizer_name: xxx
            }
        time_stamp(str): the start time of using the project
        metric(float): the value of the accuracy or other metric
        flag(str): if the cheeckpoint is final, the value of it is 'final'. else, the value of it is 'inter'
        verbose(str): some comments 
        best(bool): if the ckpt is the best one, it will be True; else it will be False
        save_model: if the engine saves the model(network), it will be True; else it will be False
    """
    # set a the lowest metric
    low = 20.0 # in order to filter the really low accuracy
    if metric < low:
        logger.info(f'|*_*| ==> Not save the model, because the low metric: {metric:.3f}')
        return {'ckpt_file': None, 'model_file': None}
    
    # create and check the output directionary
    output_ckpt = Path(cfg.TRAIN.checkpoint_output) / cfg.DATASET.name / cfg.MODEL.name /('cfg@' + cfg_name) / time_stamp
    output_ckpt.mkdir(parents=True, exist_ok=True)
    logger.info(f'=>Save the checkpoint in:{output_ckpt}')
    output_model = Path(cfg.TRAIN.model_output) / cfg.DATASET.name / cfg.MODEL.name 
    output_model.mkdir(parents=True, exist_ok=True)
    logger.info(f'=>Save the model in:{output_model}')

    saved_keys = list(saved_stuff.keys())
    # Save in a dict
    checkpoint = OrderedDict() # make the type of the checkpoint is OrderedDict
    # step = 0
    if 'step' in saved_keys:
        checkpoint['step'] = saved_stuff['step']
        # step = saved_stuff['step']
        saved_keys.remove('step')
    if 'loss' in saved_keys:
        checkpoint['loss'] = saved_stuff['loss']
        saved_keys.remove('loss')
    
    # import ipdb; ipdb.set_trace()
    for key in saved_keys:
        stuff = saved_stuff[key]
        if type(stuff) == type(dict()):
            temp = {k:v.state_dict() for k, v in stuff.items()}
            checkpoint[key] = temp
        else:
            checkpoint[key] = stuff.state_dict()
    
    # make the ckpt file name
    file_name_ckpt = f'{flag}_step{step}#{metric:.3f}^{verbose}.pth.tar'
    file_name_ckpt_best = f'best_ckpt_{cfg.DATASET.name}_{cfg.MODEL.name}#{metric:.3f}_cfg@{cfg_name}.pth'
    ckpt = output_ckpt / file_name_ckpt
    ckpt_best = Path(cfg.TRAIN.checkpoint_output) / file_name_ckpt_best

    if best:
        torch.save(checkpoint, ckpt_best)
        logger.info(f'\033[1;32m =>Save Best checkpoint:{ckpt_best} \033[0m')
        ckpt_str = str(ckpt_best)
    else:
        torch.save(checkpoint, ckpt)
        logger.info(f'\033[1;34m =>Save checkpoint:{ckpt} \033[0m')
        ckpt_str = str(ckpt)
    # return ckpt_str

    mdl_str = 'None'
    if save_model:
        # make the model name
        model_name = f'cfg@{cfg_name}#{time_stamp}#{metric:.3f}^{verbose}.pth'
        model_name_best = f'best_{cfg.DATASET.name}_{cfg.MODEL.name}#{metric:.3f}_cfg@{cfg_name}.pth'
        mdl = output_model / model_name    # mdl addr. model
        mdl_best = Path(cfg.TRAIN.model_output) / model_name_best
        if best:
            torch.save(checkpoint, mdl_best)
            logger.info(f'\033[1;31m =>Saved Best Model name:{mdl_best} \033[0m')
            mdl_str = str(mdl_best)
        else:
            torch.save(checkpoint, mdl)
            logger.info(f'\033[1;31m =>Saved Model name:{mdl} \033[0m')
            mdl_str = str(mdl)
        
    return {'ckpt_file': ckpt_str, 'model_file': mdl_str}