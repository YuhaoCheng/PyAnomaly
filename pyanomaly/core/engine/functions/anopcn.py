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

__all__ = ['ANOPCNTrainer', 'ANOPCNInference']
@ENGINE_REGISTRY.register()
class ANOPCNTrainer(BaseTrainer):
    NAME = ["ANOPCN.TRAIN"]
    def custom_setup(self):
        # basic meter
        self.loss_predmeter_G = AverageMeter(name='loss_pred_G')
        self.loss_predmeter_D = AverageMeter(name='loss_pred_D')
        self.loss_refinemeter_G = AverageMeter(name='loss_refine_G')
        self.loss_refinemeter_D = AverageMeter(name='loss_refine_D')

    def train(self,current_step):
        # Pytorch [N, C, D, H, W]
        # initialize
        self.set_requires_grad(self.F, False)
        self.set_requires_grad(self.G, True)
        self.set_requires_grad(self.D, True)
        self.D.train()
        self.G.train()
        self.F.eval()

        dynamic_steps = self.steps.param['dynamic_steps']
        temp_step = current_step % dynamic_steps[2]
        if temp_step in range(dynamic_steps[0], dynamic_steps[1]):
            self.train_pcm(current_step)
        elif temp_step in range(dynamic_steps[1], dynamic_steps[2]):
            self.train_erm(current_step)
    
    def train_pcm(self, current_step):
        # Pytorch [N, C, D, H, W]
        # initialize
        start = time.time()
        if self.kwargs['parallel']:
            self.set_requires_grad(self.G.module.pcm, True)
            self.set_requires_grad(self.G.module.erm, False)
        else:
            self.set_requires_grad(self.G.pcm, True)
            self.set_requires_grad(self.G.erm, False)
        writer = self.kwargs['writer_dict']['writer']
        global_steps = self.kwargs['writer_dict']['global_steps_{}'.format(self.kwargs['model_type'])]
        
        # get the data
        data, anno, meta = next(self._train_loader_iter)  # the core for dataloader
        self.data_time.update(time.time() - start)
        
        # base on the D to get each frame
        target = data[:, :, -1, :, :].cuda() # t frame 
        pred_last = data[:, :, -2, :, :].cuda() # t-1 frame
        input_data = data[:, :, :-1, :, :].cuda() # 0 ~ t-1 frame
        # input_data = data.cuda() # 0 ~ t frame
        
        # True Process =================Start===================
        #---------update optim_G ---------
        self.set_requires_grad(self.D, False)
        output_predframe_G, _ = self.G(input_data, target)
        
        predFlowEstim = torch.cat([pred_last, output_predframe_G],1).cuda()
        gtFlowEstim = torch.cat([pred_last, target], 1).cuda()
        gtFlow_vis, gtFlow = flow_batch_estimate(self.F, gtFlowEstim, self.normalize.param['train'], 
                                                 output_format=self.config.DATASET.optical_format, optical_size=self.config.DATASET.optical_size)
        predFlow_vis, predFlow = flow_batch_estimate(self.F, predFlowEstim, self.normalize.param['train'], 
                                                 output_format=self.config.DATASET.optical_format, optical_size=self.config.DATASET.optical_size)
        
        loss_g_adv = self.GANLoss(self.D(output_predframe_G), True)
        loss_op = self.OpticalflowSqrtLoss(predFlow, gtFlow)
        loss_int = self.IntentsityLoss(output_predframe_G, target)
        loss_gd = self.GradientLoss(output_predframe_G, target)

        loss_g_all = self.loss_lamada['IntentsityLoss'] * loss_int + self.loss_lamada['GradientLoss'] * loss_gd + \
                     self.loss_lamada['OpticalflowSqrtLoss'] * loss_op + self.loss_lamada['GANLoss'] * loss_g_adv
        self.optimizer_G.zero_grad()
        loss_g_all.backward()
        self.optimizer_G.step()
        # record
        self.loss_predmeter_G.update(loss_g_all.detach())
        if self.config.TRAIN.adversarial.scheduler.use:
            self.optimizer_G_scheduler.step()
        
        #---------update optim_D ---------------
        self.set_requires_grad(self.D, True)
        self.optimizer_D.zero_grad()
        temp_t = self.D(target)
        temp_g = self.D(output_predframe_G.detach())
        loss_d_1 = self.GANLoss(temp_t, True)
        loss_d_2 = self.GANLoss(temp_g, False)
        loss_d = (loss_d_1 + loss_d_2) * 0.5
        loss_d.backward()
        self.optimizer_D.step()
        if self.config.TRAIN.adversarial.scheduler.use:
            self.optimizer_D_scheduler.step()
        self.loss_predmeter_D.update(loss_d.detach())
        # ======================End==================

        self.batch_time.update(time.time() - start)

        if (current_step % self.steps.param['log'] == 0):
            msg = make_info_message(current_step, self.steps.param['max'], self.kwargs['model_type'], self.batch_time, 
                                    self.config.TRAIN.batch_size, self.data_time, [self.loss_predmeter_G, self.loss_predmeter_D])
            self.logger.info(msg)
        
        writer.add_scalar('Train_loss_G', self.loss_predmeter_G.val, global_steps)
        writer.add_scalar('Train_loss_D', self.loss_predmeter_D.val, global_steps)

        if (current_step % self.steps.param['vis'] == 0):
            vis_objects = OrderedDict({
                'train_target_flow': gtFlow_vis.detach(),
                'train_pred_flow': predFlow_vis.detach(),
                'train_target_frame': target.detach(),
                'train_output_predframe_G': output_predframe_G.detach()
            })
            tensorboard_vis_images(vis_objects, writer, global_steps, self.normalize.param['train'])
        
        global_steps += 1 
        # reset start
        start = time.time()
        
        self.saved_model = {'G':self.G, 'D':self.D}
        self.saved_optimizer = {'optim_G': self.optimizer_G, 'optim_D': self.optimizer_D}
        self.saved_loss = {'loss_G':self.loss_predmeter_G.val, 'loss_D':self.loss_predmeter_D.val}
        self.kwargs['writer_dict']['global_steps_{}'.format(self.kwargs['model_type'])] = global_steps

    def train_erm(self, current_step):
        # Pytorch [N, C, D, H, W]
        # initialize
        start = time.time()
        if self.kwargs['parallel']:
            self.set_requires_grad(self.G.module.erm, True)
            self.set_requires_grad(self.G.module.pcm, False)
        else:
            self.set_requires_grad(self.G.erm, True)
            self.set_requires_grad(self.G.pcm, False)
        writer = self.kwargs['writer_dict']['writer']
        global_steps = self.kwargs['writer_dict']['global_steps_{}'.format(self.kwargs['model_type'])]
        
        # get the data
        data, _ = next(self._train_loader_iter)  # the core for dataloader
        self.data_time.update(time.time() - start)
        
        # base on the D to get each frame
        target = data[:, :, -1, :, :].cuda() # t frame 
        pred_last = data[:, :, -2, :, :].cuda() # t-1 frame
        input_data = data[:, :, :-1, :, :].cuda() # 0 ~ t-1 frame
        # input_data = data.cuda() # 0 ~ t frame
        
        # True Process =================Start===================
        #---------update optim_G ---------
        self.set_requires_grad(self.D, False)
        _, output_refineframe_G = self.G(input_data, target)
        
        gtFlowEstim = torch.cat([pred_last, target], 1).cuda()
        predFlowEstim = torch.cat([pred_last, output_refineframe_G],1).cuda()

        gtFlow_vis, gtFlow = flow_batch_estimate(self.F, gtFlowEstim, self.normalize.param['train'], 
                                                 output_format=self.config.DATASET.optical_format, optical_size=self.config.DATASET.optical_size)
        predFlow_vis, predFlow = flow_batch_estimate(self.F, predFlowEstim, self.normalize.param['train'], 
                                                     output_format=self.config.DATASET.optical_format, optical_size=self.config.DATASET.optical_size)
        
        loss_g_adv = self.GANLoss(self.D(output_refineframe_G), True)
        loss_op = self.OpticalflowSqrtLoss(predFlow, gtFlow)
        loss_int = self.IntentsityLoss(output_refineframe_G, target)
        loss_gd = self.GradientLoss(output_refineframe_G, target)

        loss_g_all = self.loss_lamada['IntentsityLoss'] * loss_int + self.loss_lamada['GradientLoss'] * loss_gd + \
                     self.loss_lamada['OpticalflowSqrtLoss'] * loss_op + self.loss_lamada['GANLoss'] * loss_g_adv
        self.optimizer_G.zero_grad()
        loss_g_all.backward()
        self.optimizer_G.step()
        # record
        self.loss_refinemeter_G.update(loss_g_all.detach())
        if self.config.TRAIN.adversarial.scheduler.use:
            self.optimizer_G_scheduler.step()
        
        #---------update optim_D ---------------
        self.set_requires_grad(self.D, True)
        self.optimizer_D.zero_grad()
        temp_t = self.D(target)
        temp_g = self.D(output_refineframe_G.detach())
        loss_d_1 = self.GANLoss(temp_t, True)
        loss_d_2 = self.GANLoss(temp_g, False)
        loss_d = (loss_d_1 + loss_d_2) * 0.5
        loss_d.backward()
        self.optim_D.step()

        if self.config.TRAIN.adversarial.scheduler.use:
            self.optimizer_D_scheduler.step()
        self.loss_refinemeter_D.update(loss_d.detach())
        # ======================End==================

        self.batch_time.update(time.time() - start)

        if (current_step % self.steps.param['log'] == 0):
            msg = make_info_message(current_step, self.steps.param['max'], self.kwargs['model_type'], self.batch_time, 
                                    self.config.TRAIN.batch_size, self.data_time, [self.loss_refinemeter_G, self.loss_refinemeter_D])
            self.logger.info(msg)
        
        writer.add_scalar('Train_loss_G', self.loss_refinemeter_G.val, global_steps)
        writer.add_scalar('Train_loss_D', self.loss_refinemeter_D.val, global_steps)
        
        if (current_step % self.steps.param['vis'] == 0):
            vis_objects = OrderedDict({
                'train_target_flow': gtFlow_vis.detach(),
                'train_pred_flow': predFlow_vis.detach(),
                'train_target_frame': target.detach(),
                'train_output_refineframe_G': output_refineframe_G.detach()
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
        # self.saved_loss = {'loss_G':self.loss_refinemeter_G.val, 'loss_D':self.loss_refinemeter_D.val}
        self.saved_loss['loss_G'] = self.loss_refinemeter_G.val
        self.saved_loss['loss_D'] = self.loss_refinemeter_D.val
        self.kwargs['writer_dict']['global_steps_{}'.format(self.kwargs['model_type'])] = global_steps

@ENGINE_REGISTRY.register()
class ANOPCNInference(BaseInference):
    NAME = ["ANOPCN.INFERENCE"]
    def inference(self):
        for h in self._hooks:
            h.inference()
    