"""
@author:  Yuhao Cheng
@contact: yuhao.cheng[at]outlook.com
"""
from .hook_registry import HOOK_REGISTRY
from .functions import *
import logging
logger = logging.getLogger(__name__)

class HookAPI(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.train_hook_names = cfg.MODEL.hooks.train
        self.val_hook_names = cfg.MODEL.hooks.val

    def __call__(self, is_training):
        if is_training:
            mode = 'train'
            hook_names = self.train_hook_names
        else:
            mode = 'val/test'
            hook_names = self.val_hook_names
        
        logger.info(f'{mode}*********use hooks:{hook_names}**********')
        hooks = []
        for name in hook_names:
            temp = HOOK_REGISTRY.get(name)()
            hooks.append(temp)
        logger.info(f'build:{hooks}')
        return hooks

