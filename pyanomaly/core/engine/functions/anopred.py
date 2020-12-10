"""
@author:  Yuhao Cheng
@contact: yuhao.cheng[at]outlook.com
"""
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
import torchvision.transforms.functional as tf
from torch.utils.data import DataLoader
# torch.autograd.set_detect_anomaly(True)

from pyanomaly.core.utils import AverageMeter, flow_batch_estimate, tensorboard_vis_images, make_info_message, ParamSet
# from pyanomaly.datatools.evaluate.utils import psnr_error
from ..abstract.default_engine import DefaultTrainer, DefaultInference

from ..engine_registry import ENGINE_REGISTRY

__all__ = ['ANOPREDTrainer', 'ANOPREDInference']

@ENGINE_REGISTRY.register()
class ANOPREDTrainer(DefaultTrainer):
    NAME = ["ANOPRED.TRAIN"]
    def custom_setup(self):
        # basic meter
        self.loss_meter_G = AverageMeter(name='Loss_G')
        self.loss_meter_D = AverageMeter(name='Loss_D')

        # others
        self.optical = ParamSet(name='optical', size=self.config.DATASET.optical_size, output_format=self.config.DATASET.optical_format)

    
    def train(self,current_step):
        # Pytorch [N, C, D, H, W]
        # initialize
        start = time.time()
        self.set_requires_grad(self.F, False)
        self.set_requires_grad(self.G, True)
        self.set_requires_grad(self.D, True)
        self.G.train()
        self.D.train()
        self.F.eval()
        writer = self.kwargs['writer_dict']['writer']
        global_steps = self.kwargs['writer_dict']['global_steps_{}'.format(self.kwargs['model_type'])]

        # get the data
        data, anno, meta = next(self._train_loader_iter) 
        self.data_time.update(time.time() - start)

        # base on the D to get each frame
        target = data[:, :, -1, :, :].cuda() # t+1 frame 
        input_data = data[:, :, :-1, :, :] # 0 ~ t frame
        input_last = input_data[:, :, -1, :, :].cuda() # t frame

        # squeeze the D dimension to C dimension, shape comes to [N, C, H, W]
        input_data = input_data.reshape(input_data.shape[0], -1, input_data.shape[-2], input_data.shape[-1]).cuda()

        # True Process =================Start===================
        #---------update optim_G ---------
        self.set_requires_grad(self.D, False)
        output_pred_G = self.G(input_data)
        # import ipdb; ipdb.set_trace()
        predFlowEstim = torch.cat([input_last, output_pred_G],1)
        gtFlowEstim = torch.cat([input_last, target], 1)
        gtFlow_vis, gtFlow = flow_batch_estimate(self.F, gtFlowEstim, self.normalize.param['train'], output_format=self.config.DATASET.optical_format, optical_size=self.config.DATASET.optical_size)
        predFlow_vis, predFlow = flow_batch_estimate(self.F, predFlowEstim, self.normalize.param['train'], output_format=self.config.DATASET.optical_format, optical_size=self.config.DATASET.optical_size)

        loss_g_adv = self.GANLoss(self.D(output_pred_G), True)
        loss_op = self.OpticalflowLoss(predFlow, gtFlow)
        loss_int = self.IntentsityLoss(output_pred_G, target)
        loss_gd = self.GradientLoss(output_pred_G, target)
        loss_g_all = self.loss_lamada['IntentsityLoss'] * loss_int + self.loss_lamada['GradientLoss'] * loss_gd + self.loss_lamada['OpticalflowLoss'] * loss_op + self.loss_lamada['GANLoss'] * loss_g_adv
        self.optimizer_G.zero_grad()
        loss_g_all.backward()
        self.optimizer_G.step()

        # record
        self.loss_meter_G.update(loss_g_all.detach())
        
        if self.config.TRAIN.adversarial.scheduler.use:
            self.optimizer_G_scheduler.step()
        #---------update optim_D ---------------
        self.set_requires_grad(self.D, True)
        self.optimizer_D.zero_grad()
        # G_output = self.G(input)
        temp_t = self.D(target)
        temp_g = self.D(output_pred_G.detach())
        loss_d_1 = self.GANLoss(temp_t, True)
        loss_d_2 = self.GANLoss(temp_g, False)
        loss_d = (loss_d_1 + loss_d_2) * 0.5
        # loss_d.sum().backward()
        loss_d.backward()

        self.optimizer_D.step()
        if self.config.TRAIN.adversarial.scheduler.use:
            self.optimizer_D_scheduler.step()
        self.loss_meter_D.update(loss_d.detach())
        # ======================End==================

        self.batch_time.update(time.time() - start)

        if (current_step % self.steps.param['log'] == 0):
            msg = make_info_message(current_step, self.steps.param['max'], self.kwargs['model_type'], self.batch_time, 
                                    self.config.TRAIN.batch_size, self.data_time, [self.loss_meter_G, self.loss_meter_D])
            self.logger.info(msg)
        
        writer.add_scalar('Train_loss_G', self.loss_meter_G.val, global_steps)
        writer.add_scalar('Train_loss_D', self.loss_meter_D.val, global_steps)
        
        if (current_step % self.steps.param['vis'] == 0):
            vis_objects = OrderedDict({
                'train_target': target.detach(),
                'train_output_pred_G': output_pred_G.detach(),
                'train_gtFlow': gtFlow_vis.detach(),
                'train_predFlow': predFlow_vis.detach()
            })
            tensorboard_vis_images(vis_objects, writer, global_steps, self.normalize.param['train'])
        
        global_steps += 1 
        # reset start
        start = time.time()
        
        # self.saved_model = {'G':self.G, 'D':self.D}
        self.saved_model['G'] = self.G
        self.saved_model['D'] = self.D
        # self.saved_optimizer = {'optim_G': self.optimizer_G, 'optim_D': self.optimizer_D}
        self.saved_optimizer['optimizer_G'] = self.optimizer_G
        self.saved_optimizer['optimizer_D'] = self.optimizer_D
        # self.saved_loss = {'loss_G':self.loss_meter_G.val, 'loss_D':self.loss_meter_D.val}
        self.saved_loss['loss_G'] = self.loss_meter_G.val
        self.saved_loss['loss_D'] = self.loss_meter_D.val
        self.kwargs['writer_dict']['global_steps_{}'.format(self.kwargs['model_type'])] = global_steps
    
@ENGINE_REGISTRY.register()
class ANOPREDInference(DefaultInference):
    NAME = ["ANOPRED.INFERENCE"]    
    def inference(self):
        for h in self._hooks:
            h.inference()