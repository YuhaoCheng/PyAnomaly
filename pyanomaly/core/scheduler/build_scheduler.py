import torch
from .schedulers import WarmupMultiStepLR, WarmupCosineLR
from collections import OrderedDict
class SchedulerAPI(object):
    _SUPPORT = ['stepLR', 'cosLR', 'WarmupCosLR', 'WarmupMultiStepLR']
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.train_mode = cfg.TRAIN.mode
        if self.train_mode == 'general':
            self.type = self.cfg.TRAIN.general.scheduler.name
            self.params = self.cfg.TRAIN.general.scheduler
        elif self.train_mode == 'adversarial':
            self.type = self.cfg.TRAIN.adversarial.scheduler.name
            self.params = self.cfg.TRAIN.adversarial.scheduler
        
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
        return t_scheduelr
    
    def __call__(self, optim_dict):
        scheduler_dict = OrderedDict()
        optim_names = optim_dict.keys()
        for name in optim_names:
            one_scheduler = self._build_scheduler(optim_dict[name])
            scheduler_dict.update({f'{name}_scheduler': one_scheduler})
        
        return scheduler_dict
        
        
