"""
@author:  Yuhao Cheng
@contact: yuhao.cheng[at]outlook.com
"""
from collections import OrderedDict
from .functions import *
from .loss_registry import LOSS_REGISTRY
import logging
logger = logging.getLogger(__name__)

class LossAPI(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.losses = self.cfg.TRAIN.losses
        assert len(self.losses) % 4 == 0, 'The lengths of the losses must the 4 times'
        self._individual_losses = [self.losses[i:i+4] for i in range(0, len(self.losses), 4)]
        # import ipdb; ipdb.set_trace()
        # self.logger = logger
    
    def build(self):
        loss_dict = OrderedDict()
        loss_coefficient_dict = OrderedDict()
        for index, couple in enumerate(self._individual_losses):
            register_name, loss_name, loss_devicetype = couple[0].split('_')
            loss_cfg = couple[3]
            # decide the register 
            if register_name == 'loss':
                # import ipdb; ipdb.set_trace()
                if len(loss_cfg) != 0:
                    loss_dict[loss_name] = LOSS_REGISTRY.get(couple[2])(loss_cfg=loss_cfg)
                else:
                    print(loss_name)
                    loss_dict[loss_name] = LOSS_REGISTRY.get(couple[2])()
            else:
                raise Exception(f'The name of {register_name} is not supported')
            
            # change the device type 
            if loss_devicetype == 'cuda':
                loss_dict[loss_name] = loss_dict[loss_name].cuda() 
            loss_coefficient_dict[loss_name] = couple[1]
        return loss_dict, loss_coefficient_dict
    
    def __call__(self):
        
        loss_dict, loss_coefficient_dict = self.build()
        return loss_dict, loss_coefficient_dict