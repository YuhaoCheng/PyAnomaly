# from .abstract.abstract_loss_builder import LossBuilder
from collections import OrderedDict
from .functions import *
from .loss_registry import LOSS_REGISTRY

# class LossBuilder(object):
#     def __init__(self, cfg):
#         self.cfg = cfg
#         self.loss_list = self.cfg.TRAIN.loss
#         self.loss_coefficient_list = self.cfg.TRAIN.loss_coefficients
#         assert len(self.loss_list) == len(self.loss_coefficient_list), 'The lengths of the loss and the coefficient are not equal!!'
    
#     def _get_loss(self, loss_name, cfg):
#         if loss_name in EXPAND_LOSS:
#             return get_expand_loss(loss_name, cfg)
#         else:
#             return get_basic_loss(loss_name, cfg)

#     def build(self):
#         loss_dict = OrderedDict()
#         loss_coefficient_dict = OrderedDict()
#         for index,loss_name in enumerate(self.loss_list):
#             loss_dict[loss_name] = self._get_loss(loss_name, self.cfg)
#             loss_coefficient_dict[loss_name] = self.loss_coefficient_list[index]

#         return loss_dict, loss_coefficient_dict

# class LossAPI(LossBuilder):
#     def __init__(self, cfg, logger):
#         super(LossAPI, self).__init__(cfg)
#         self.logger = logger
#     def __call__(self):
#         loss_dict, loss_lamada = super(LossAPI, self).build()
#         self.logger.info(f'the loss names:{loss_dict.keys()}')
#         self.logger.info(f'the loss lamada:{loss_lamada}')
#         return loss_dict, loss_lamada

# class LossAPI(LossBuilder):
class LossAPI(object):
    def __init__(self, cfg, logger):
        # super(LossAPI, self).__init__(cfg)
        self.cfg = cfg
        # self.loss_list = self.cfg.TRAIN.loss
        # self.loss_coefficient_list = self.cfg.TRAIN.loss_coefficients
        self.losses = self.cfg.TRAIN.losses
        assert len(self.losses) % 4 == 0, 'The lengths of the losses must the 4 times'
        self._individual_losses = [self.losses[i:i+4] for i in range(0, len(self.losses), 4)]
        # import ipdb; ipdb.set_trace()
        self.logger = logger
    
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
                    loss_dict[loss_name] = LOSS_REGISTRY.get(couple[2])(loss_cfg)
                else:
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