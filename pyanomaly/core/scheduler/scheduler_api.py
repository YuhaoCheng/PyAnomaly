import torch
from .schedulers import WarmupMultiStepLR, WarmupCosineLR
from collections import OrderedDict
import logging 
logger = logging.getLogger(__name__)

class SchedulerAPI(object):
    _SUPPORT = ['stepLR', 'cosLR', 'WarmupCosLR', 'WarmupMultiStepLR', 'MultiStepLR']
    def __init__(self, cfg):
        self.cfg = cfg
        # self.logger = logger
        self.train_mode = cfg.TRAIN.mode
        self.params = self.cfg.get('TRAIN')[self.train_mode]['scheduler']
        self.type = self.params.name
        
    def _build_scheduler(self, optimizer_param):
        if self.type not in SchedulerAPI._SUPPORT:
            raise Exception(f'No support: {self.type} in the SchedulerAPI')
        elif self.type == SchedulerAPI._SUPPORT[0]:
            t_scheduelr = torch.optim.lr_scheduler.StepLR(optimizer_param, self.params.step_size, self.params.gamma)
        elif self.type == SchedulerAPI._SUPPORT[1]:
            t_scheduelr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_param, self.params.T_max,self.params.eta_min)
        elif self.type == SchedulerAPI._SUPPORT[2]:
            t_scheduelr = WarmupCosineLR(optimizer_param, self.params.T_max, warmup_factor=self.params.warmup_factor, warmup_iters=self.params.warmup_iters, warmup_method=self.params.warmup_method)
        elif self.type == SchedulerAPI._SUPPORT[3]:
            t_scheduelr = WarmupMultiStepLR(optimizer_param, self.params.steps, self.params.gamma, warmup_factor=self.params.warmup_factor, warmup_iters=self.params.warmup_iters, warmup_method=self.params.warmup_method)
        elif self.type == SchedulerAPI._SUPPORT[4]:
            t_scheduelr = torch.optim.lr_scheduler.MultiStepLR(optimizer_param, self.params.steps, gamma=self.params.gamma)
        else:
            raise Exception(f'Not Support Scheduler! The support schedulers are {SchedulerAPI._SUPPORT}')
        return t_scheduelr
    
    def __call__(self, optim_dict):
        scheduler_dict = OrderedDict()
        optim_names = optim_dict.keys()
        for name in optim_names:
            one_scheduler = self._build_scheduler(optim_dict[name])
            scheduler_dict.update({f'{name}_scheduler': one_scheduler})
        
        return scheduler_dict
        
        
