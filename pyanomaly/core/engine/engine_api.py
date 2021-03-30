"""
@author:  Yuhao Cheng
@contact: yuhao.cheng[at]outlook.com
"""
from .engine_registry import ENGINE_REGISTRY
from .functions import *
import logging
logger = logging.getLogger(__name__)

class EngineAPI(object):
    '''
    The  API class to get engines which are used in training or inference process
    '''
    def __init__(self, cfg, is_training):
        '''
        The initialization method of the EngineAPI 
        Args:
            cfg: The config class
            is_training: indicate whether the engine is used in training process
        Return:
            None
        '''
        self.cfg = cfg
        self.model_name = self.cfg.MODEL.name
        self.is_training = is_training
        self.phase = 'TRAIN' if self.is_training else 'VAL'
        self.engine_name = self.cfg.get(self.phase)['engine_name']
    
    def build(self):
        '''
        The method to produce the engine
        Args:
            None
        Returns:
            engine: The Engine class which is used for training or inference
        '''
        engine = ENGINE_REGISTRY.get(self.engine_name)

        logger.info(f'{self.model_name} use the engine: {self.engine_name} in phase of {self.phase}')

        return engine


