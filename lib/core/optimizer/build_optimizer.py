import torch
import torch.optim as optim
from collections import OrderedDict
class OptimizerAPI(object):
    _SUPPROT = ['adam', 'sgd']
    _NAME = 'OptimizerAPI'
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.train_mode = cfg.TRAIN.mode
        if self.train_mode == 'general':
            self.type = self.cfg.TRAIN.general.optimizer.name
            self.params = self.cfg.TRAIN.general.optimizer
            self.lr = self.params.lr
        elif self.train_mode == 'adversarial':
            self.type = self.cfg.TRAIN.adversarial.optimizer.name
            self.params = self.cfg.TRAIN.adversarial.optimizer
            self.lr = 0        
        self.setup()

    def setup(self):
        self.lr_dict = OrderedDict()
        if self.train_mode == 'adversarial':
            self.lr_dict.update({'Generator':self.params.g_lr})
            self.lr_dict.update({'Discriminator':self.params.d_lr})

    def update(self, new_lr, verbose='none'):
        old_lr = self.lr
        self.lr = new_lr
        self.logger.info(f'{verbose}Upate the LR from {old_lr} to {self.lr}')

    def _build_one_optimizer(self, model):
        if self.type not in OptimizerAPI._SUPPROT:
            raise Exception(f'Not support: {self.type} in {OptimizerAPI._NAME}')
        elif self.type == 'adam':
            t = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.params.weight_decay)
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
            t = torch.optim.Adam(param_groups, lr=self.lr, weight_decay=self.params.weight_decay)
        elif self.type == 'sgd':
            for model in model_list:
                param_groups.append({'params':model.parameters()}) 
            t = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.params.momentum, weight_decay=self.params.weight_decay,nesterov=self.params.nesterov)
        
        return t
    
    def _build(self, model):
        if isinstance(model, torch.nn.Module):
            self.logger.info('Build the optimizer for one model')
            opimizer = self._build_one_optimizer(model)
        elif isinstance(model, list):
            self.logger.info('Build the optimizer for multi models')
            opimizer = self._build_multi_optimizers(model)
        return opimizer
    
    def __call__(self, model):
        include_parts = self.params.include
        output_names = self.params.output_name
        assert len(include_parts) >= len(output_names), f'Not support the situation: the number of model part ({len(include_parts)}) > the number of output optimizer ({len(output_names)})'
        self.logger.info(f'=> Build the optimzer of {self.train_mode} include {include_parts}')
        t_dict = OrderedDict()

        if len(output_names) == 1:
            model_combination = []
            for temp in include_parts:
                model_combination.append(model[temp])
            t = self._build(model_combination)
            t_dict.update({output_names[0]:t})
            # return t_dict
        elif len(output_names) == len(include_parts):
            for index, temp in enumerate(include_parts):
                self.update(self.lr_dict[temp], str(temp))
                t = self._build(model[temp])
                t_dict.update({output_names[index]:t})
        else:
            raise Exception(f'Not support the situation: the number of model part ({len(include_parts)}) and  the number of output optimizer ({len(output_names)})')
        return t_dict

