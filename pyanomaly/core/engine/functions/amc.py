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

from pyanomaly.core.utils import AverageMeter, flow_batch_estimate, tensorboard_vis_images, vis_optical_flow, make_info_message, ParamSet
from pyanomaly.datatools.evaluate.utils import psnr_error
from ..abstract.default_engine import DefaultTrainer, DefaultInference

from ..engine_registry import ENGINE_REGISTRY

__all__ = ['AMCTrainer', 'AMCInference']

@ENGINE_REGISTRY.register()
class AMCTrainer(DefaultTrainer):
    NAME = ["AMC.TRAIN"]    
    def custom_setup(self):
        # create loss meters
        self.loss_meter_G = AverageMeter(name='Loss_G')
        self.loss_meter_D = AverageMeter(name='Loss_D')

        self.optical = ParamSet(name='optical', size=self.config.DATASET.optical_size, output_format=self.config.DATASET.optical_format)
        # import ipdb; ipdb.set_trace()
    
    def train(self,current_step):
        # Pytorch [N, C, D, H, W]
        # initialize
        start = time.time()
        self.set_requires_grad(self.F, False)
        self.set_requires_grad(self.D, True)
        self.set_requires_grad(self.G, True)
        self.G.train()
        self.D.train()
        self.F.eval()
        writer = self.kwargs['writer_dict']['writer']
        global_steps = self.kwargs['writer_dict']['global_steps_{}'.format(self.kwargs['model_type'])]
        
        # get the data
        data, anno, meta = next(self._train_loader_iter)
        self.data_time.update(time.time() - start)
        
        # base on the D to get each frame
        # in this method, D = 2 and not change
        input_data = data[:, :, 0, :, :].cuda() # input(1-st) frame
        target = data[:, :, 1,:, :].cuda() # target(2-nd) frame 
        
        # True Process =================Start===================
        #---------update optim_G ---------
        self.set_requires_grad(self.D, False)
        output_flow_G,  output_frame_G = self.G(input_data)
        gt_flow_esti_tensor = torch.cat([input_data, target], 1)
        # gt_flow_esti_tensor = torch.cat([input_data_original, target_original], 1)
        # flow_gt, _ = flow_batch_estimate(self.F, gt_flow_esti_tensor, optical_size=self.config.DATASET.optical_size, output_format=self.config.DATASET.optical_format, normalize=self.config.ARGUMENT.train.normal.use, mean=self.config.ARGUMENT.train.normal.mean, std=self.config.ARGUMENT.train.normal.std)
        flow_gt_vis, flow_gt  = flow_batch_estimate(self.F, gt_flow_esti_tensor, self.normalize.param['train'],
                                                    optical_size=self.config.DATASET.optical_size, output_format=self.config.DATASET.optical_format)
        fake_g = self.D(torch.cat([target, output_flow_G], dim=1))

        # loss_g_adv = self.gan_loss(fake_g, True)
        # loss_op = self.op_loss(output_flow_G, flow_gt)
        # loss_int = self.int_loss(output_frame_G, target)
        # loss_gd = self.gd_loss(output_frame_G, target)
        # loss_g_all = self.loss_lamada['intentsity_loss'] * loss_int + self.loss_lamada['gradient_loss'] * loss_gd + self.loss_lamada['opticalflow_loss'] * loss_op + self.loss_lamada['gan_loss'] * loss_g_adv

        loss_g_adv = self.GANLoss(fake_g, True)
        loss_op = self.OpticalflowSqrtLoss(output_flow_G, flow_gt)
        loss_int = self.IntentsityLoss(output_frame_G, target)
        loss_gd = self.GradientLoss(output_frame_G, target)
        loss_g_all = self.loss_lamada['IntentsityLoss'] * loss_int + self.loss_lamada['GradientLoss'] * loss_gd + self.loss_lamada['OpticalflowSqrtLoss'] * loss_op + self.loss_lamada['GANLoss'] * loss_g_adv

        self.optimizer_G.zero_grad()
        loss_g_all.backward()
        self.optimizer_G.step()
        self.loss_meter_G.update(loss_g_all.detach())
        
        if self.config.TRAIN.adversarial.scheduler.use:
            self.optimizer_G_scheduler.step()

        #---------update optim_D ---------------
        self.set_requires_grad(self.D, True)
        self.optimizer_D.zero_grad()
        # import ipdb; ipdb.set_trace()
        real_d = self.D(torch.cat([target, flow_gt],dim=1))
        fake_d = self.D(torch.cat([target, output_flow_G.detach()], dim=1))
        # loss_d_1 = self.gan_loss(real_d, True)
        # loss_d_2 = self.gan_loss(fake_d, False)
        loss_d_1 = self.GANLoss(real_d, True)
        loss_d_2 = self.GANLoss(fake_d, False)
        loss_d = (loss_d_1  + loss_d_2) * 0.5 
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
            temp = vis_optical_flow(output_flow_G.detach(), output_format=self.config.DATASET.optical_format, output_size=(output_flow_G.shape[-2], output_flow_G.shape[-1]), 
                                    normalize=self.normalize.param['train'])
            vis_objects = OrderedDict({
                'train_target_flow': flow_gt_vis.detach(),
                'train_output_flow_G': temp, 
                'train_target_frame': target.detach(),
                'train_output_frame_G': output_frame_G.detach(),
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
class AMCInference(DefaultInference):
    NAME = ["AMC.INFERENCE"]

    def inference(self):
        for h in self._hooks:
            h.inference()
    