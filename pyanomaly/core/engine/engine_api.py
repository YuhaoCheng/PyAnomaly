"""
@author:  Yuhao Cheng
@contact: yuhao.cheng[at]outlook.com
"""
from .engine_registry import ENGINE_REGISTRY
from .functions import *
import logging
logger = logging.getLogger(__name__)

class EngineAPI(object):
    def __init__(self, cfg, is_training):
        self.cfg = cfg
        # self.train_hook_names = cfg.MODEL.hooks.train
        # self.val_hook_names = cfg.MODEL.hooks.val
        # self.eval_hook_names = cfg.MODEL.eval_hooks
        # import ipdb; ipdb.set_trace()
        self.model_name = self.cfg.MODEL.name
        self.is_training = is_training
        if self.is_training:
            self.phase = 'TRAIN'
        else:
            self.phase = 'VAL'

        self.engine_name = self.cfg.get(self.phase)['engine_name']
    
    def build(self):
        engine = ENGINE_REGISTRY.get(self.engine_name)

        logger.info(f'{self.model_name} use the engine: {self.engine_name} in phase of {self.phase}')

        return engine


