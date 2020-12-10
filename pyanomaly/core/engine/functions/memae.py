"""
@author:  Yuhao Cheng
@contact: yuhao.cheng[at]outlook.com
"""
#!!!!! ignore the warning messages
import warnings
warnings.filterwarnings('ignore')
import os
import pickle
import math
import torch
import time
import numpy as np
from PIL import Image
from collections import OrderedDict
import torchvision.transforms as T
import torchvision.transforms.functional as tf
from torch.utils.data import DataLoader

from pyanomaly.core.utils import AverageMeter, flow_batch_estimate, tensorboard_vis_images, make_info_message, ParamSet
from pyanomaly.datatools.evaluate.utils import psnr_error
from ..abstract.base_engine import BaseTrainer, BaseInference
from ..engine_registry import ENGINE_REGISTRY

__all__ = ['MEMAETrainer', 'MEMAEInference']

@ENGINE_REGISTRY.register()
class MEMAETrainer(BaseTrainer):
    NAME = ["MEMAE.TRAIN"]
    def custom_setup(self):

        # basic meter
        self.loss_meter_MemAE = AverageMeter(name='loss_memae')

    
    def train(self,current_step):
        # Pytorch [N, C, D, H, W]
        # initialize
        start = time.time()
        self.set_requires_grad(self.MemAE, True)
        self.MemAE.train()
        writer = self.kwargs['writer_dict']['writer']
        global_steps = self.kwargs['writer_dict']['global_steps_{}'.format(self.kwargs['model_type'])]
        
        # get the data
        data, anno, meta = next(self._train_loader_iter)  # the core for dataloader
        self.data_time.update(time.time() - start)
        
        input_data = data.cuda() 
        
        # True Process =================Start===================
        output_rec, att = self.MemAE(input_data)
        loss_rec = self.rec_loss(output_rec, input_data)
        loss_mem = self.mem_loss(att)
        loss_memae_all = self.loss_lamada['rec_loss'] * loss_rec + self.loss_lamada['mem_loss'] * loss_mem 
        # loss_memae_all = self.loss_lamada['rec_loss'] * loss_rec 
        self.optim_MemAE.zero_grad()
        # with torch.autograd.set_detect_anomaly(True):
        loss_memae_all.backward()
        self.optim_MemAE.step()
        self.loss_meter_MemAE.update(loss_memae_all.detach())
        
        if self.config.TRAIN.adversarial.scheduler.use:
            self.lr_memae.step()
        # ======================End==================

        self.batch_time.update(time.time() - start)

        if (current_step % self.steps.param['log'] == 0):
            msg = make_info_message(current_step, self.steps.param['max'], self.kwargs['model_type'], self.batch_time, 
                                    self.config.TRAIN.batch_size, self.data_time, [self.loss_meter_MemAE])
            self.logger.info(msg)
        writer.add_scalar('Train_loss_MemAE', self.loss_meter_MemAE.val, global_steps)

        if (current_step % self.steps.param['vis'] == 0):
            vis_objects = OrderedDict({
                'train_output_rec_memeae': output_rec.detach(),
                'train_input':  input_data.detach()
            })
            tensorboard_vis_images(vis_objects, writer, global_steps, self.normalize.param['train'])
        global_steps += 1 
        
        # reset start
        start = time.time()
        
        # self.saved_model = {'MemAE':self.MemAE}
        self.saved_model['MemAE'] = self.MemAE
        # self.saved_optimizer = {'optim_MemAE': self.optim_MemAE}
        self.saved_optimizer['optimizer_MemAE']= self.optim_MemAE
        # self.saved_loss = {'loss_MemAE':self.loss_meter_MemAE.val}
        self.saved_loss['loss_MemAE'] = self.loss_meter_MemAE.val
        self.kwargs['writer_dict']['global_steps_{}'.format(self.kwargs['model_type'])] = global_steps


@ENGINE_REGISTRY.register()
class MEMAEInference(BaseInference):
    NAME = ["MEMAE.INFERENCE"]
    def inference(self):
        for h in self._hooks:
            h.inference()