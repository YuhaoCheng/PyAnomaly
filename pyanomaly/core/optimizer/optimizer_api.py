"""
@author:  Yuhao Cheng
@contact: yuhao.cheng[at]outlook.com
"""
import torch
import torch.optim as optim
from collections import OrderedDict
import logging
logger  = logging.getLogger(__name__)

class OptimizerAPI(object):
    _SUPPROT = ['adam', 'sgd']
    _MODE = ['all', 'individual']
    _NAME = 'OptimizerAPI'
    def __init__(self, cfg):
        self.cfg = cfg
        self.train_mode = cfg.TRAIN.mode
        self.params = self.cfg.get('TRAIN')[self.train_mode]['optimizer']
        self.type = self.params.name
        self.lrs = list(map(float,self.params.lrs))
        self.lr = self.lrs[0]

        self.setup()

    def setup(self):
        self.lr_dict = OrderedDict()
        if self.train_mode == 'adversarial':
            # self.lr_dict.update({'Generator':self.params.g_lr})
            # self.lr_dict.update({'Discriminator':self.params.d_lr})
            self.lr_dict.update({'Generator':self.lrs[0]})
            self.lr_dict.update({'Discriminator':self.lrs[1]})

    def update(self, new_lr, verbose='none'):
        old_lr = self.lr
        self.lr = new_lr
        logger.info(f'{verbose} Upate the LR from {old_lr} to {self.lr}')

    def _build_one_optimizer(self, model):
        if self.type not in OptimizerAPI._SUPPROT:
            raise Exception(f'Not support: {self.type} in {OptimizerAPI._NAME}')
        elif self.type == 'adam':
            t = torch.optim.Adam(model.parameters(), lr=self.lr, betas=self.params.betas, weight_decay=self.params.weight_decay)
        elif self.type == 'sgd':
            t = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.params.momentum, weight_decay=self.params.weight_decay,nesterov=self.params.nesterov)

        return t
    
    def _build_multi_optimizers(self, model_list):
        param_groups = list()

        if self.type not in OptimizerAPI._SUPPROT:
            raise Exception(f'Not support: {self.type} in {OptimizerAPI._NAME}')
        elif self.type == 'adam':
            for model in model_list:
                param_groups.append({'params':model.parameters()}) 
            t = torch.optim.Adam(param_groups, lr=self.lr, betas=self.params.betas, weight_decay=self.params.weight_decay)
        elif self.type == 'sgd':
            for model in model_list:
                param_groups.append({'params':model.parameters()}) 
            t = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.params.momentum, weight_decay=self.params.weight_decay,nesterov=self.params.nesterov)
        
        return t
    
    def _build(self, model):
        if isinstance(model, torch.nn.Module):
            logger.info('Build the optimizer for one model')
            opimizer = self._build_one_optimizer(model)
        elif isinstance(model, list):
            logger.info('Build the optimizer for multi models')
            opimizer = self._build_multi_optimizers(model)
        else:
            raise Exception('The mode type is not supported!')
        return opimizer
    
    def __call__(self, model):
        include_parts = self.params.include
        mode = self.params.mode
        logger.info(f'=> Build the optimizer of {self.train_mode} include {include_parts}')

        optimizer_dict = OrderedDict()

        if mode == OptimizerAPI._MODE[0]:
            optimizer_name = 'optimizer'+'_'.join(include_parts)
            model_combination = []
            for temp in include_parts:
                model_combination.append(model[temp])
            optimizer_value = self._build(model_combination)
            optimizer_dict.update({optimizer_name:optimizer_value})
        elif mode == OptimizerAPI._MODE[1]:
            for index, temp in enumerate(include_parts):
                optimizer_name = f'optimizer_{temp}'
                self.update(self.lrs[index], str(temp))
                optimizer_value = self._build(model[temp])
                optimizer_dict.update({optimizer_name:optimizer_value})
        else:
            raise Exception(f'Not support the optimizer mode, only support {OptimizerAPI._MODE}')
        
        return optimizer_dict

