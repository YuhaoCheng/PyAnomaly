"""
@author:  Yuhao Cheng
@contact: yuhao.cheng[at]outlook.com
"""
import sys
sys.path.append('../')
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
torch.autograd.set_detect_anomaly(True)

from pyanomaly.core.utils import AverageMeter, flow_batch_estimate, tensorboard_vis_images, make_info_message, ParamSet
from pyanomaly.datatools.evaluate.utils import psnr_error
from ..abstract.default_engine import DefaultTrainer, DefaultInference

from ..engine_registry import ENGINE_REGISTRY

__all__ = ['ANOPREDTrainer', 'ANOPREDInference']

@ENGINE_REGISTRY.register()
class ANOPREDTrainer(DefaultTrainer):
    NAME = ["ANOPRED.TRAIN"]
    def custom_setup(self):
        # # basic things
        # if self.kwargs['parallel']:
        #     self.G = self.data_parallel(self.model['Generator'])
        #     self.D = self.data_parallel(self.model['Discriminator'])
        #     self.F = self.data_parallel(self.model['FlowNet'])
        # else:
        #     self.G = self.model['Generator'].cuda()
        #     self.D = self.model['Discriminator'].cuda()
        #     self.F = self.model['FlowNet'].cuda() # lite flownet
        
        # # get the optimizer
        # self.optim_G = self.optimizer['optimizer_g']
        # self.optim_D = self.optimizer['optimizer_d']

        # get the loss_fucntion
        # self.gan_loss = self.loss_function['gan_loss_mse'] #'GANLoss'
        # self.gd_loss = self.loss_function['gradient_loss'] #GradientLoss
        # self.int_loss = self.loss_function['intentsity_loss']
        # self.op_loss = self.loss_function['opticalflow_loss']

        # the lr scheduler
        # self.lr_g = self.lr_scheduler_dict['optimizer_g_scheduler']
        # self.lr_d = self.lr_scheduler_dict['optimizer_d_scheduler']

        # basic meter
        self.loss_meter_G = AverageMeter(name='Loss_G')
        self.loss_meter_D = AverageMeter(name='Loss_D')
        # self.psnr = AverageMeter()

        # others
        self.optical = ParamSet(name='optical', size=self.config.DATASET.optical_size, output_format=self.config.DATASET.optical_format)

        # self.test_dataset_keys = self.kwargs['test_dataset_keys']
        # self.test_dataset_dict = self.kwargs['test_dataset_dict']

    
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

        # loss_g_adv = self.g_adv_loss(self.D(G_output))
        loss_g_adv = self.gan_loss(self.D(output_pred_G), True)
        loss_op = self.op_loss(predFlow, gtFlow)
        loss_int = self.int_loss(output_pred_G, target)
        loss_gd = self.gd_loss(output_pred_G, target)
        loss_g_all = self.loss_lamada['intentsity_loss'] * loss_int + self.loss_lamada['gradient_loss'] * loss_gd + self.loss_lamada['opticalflow_loss'] * loss_op + self.loss_lamada['gan_loss_mse'] * loss_g_adv
        self.optim_G.zero_grad()
        loss_g_all.backward()
        self.optim_G.step()
        # record
        self.loss_meter_G.update(loss_g_all.detach())
        
        if self.config.TRAIN.adversarial.scheduler.use:
            self.lr_g.step()
        #---------update optim_D ---------------
        self.set_requires_grad(self.D, True)
        self.optim_D.zero_grad()
        # G_output = self.G(input)
        temp_t = self.D(target)
        temp_g = self.D(output_pred_G.detach())
        loss_d_1 = self.gan_loss(temp_t, True)
        loss_d_2 = self.gan_loss(temp_g, False)
        loss_d = (loss_d_1 + loss_d_2) * 0.5
        # loss_d.sum().backward()
        loss_d.backward()

        self.optim_D.step()
        if self.config.TRAIN.adversarial.scheduler.use:
            self.lr_d.step()
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
        
        self.saved_model = {'G':self.G, 'D':self.D}
        self.saved_optimizer = {'optim_G': self.optim_G, 'optim_D': self.optim_D}
        self.saved_loss = {'loss_G':self.loss_meter_G.val, 'loss_D':self.loss_meter_D.val}
        self.kwargs['writer_dict']['global_steps_{}'.format(self.kwargs['model_type'])] = global_steps
    
    def mini_eval(self, current_step):
        if current_step % self.config.TRAIN.mini_eval_step != 0:
            return
        temp_meter_frame = AverageMeter()
        temp_meter_flow = AverageMeter()
        self.set_requires_grad(self.G, False)
        self.set_requires_grad(self.D, False)
        self.set_requires_grad(self.F, False)
        self.G.eval()
        self.D.eval()
        self.F.eval()
        for data, _ in self.val_dataloader:
            # base on the D to get each frame
            target_mini = data[:, :, -1, :, :].cuda() # t+1 frame 
            input_data = data[:, :, :-1, :, :] # 0 ~ t frame
            input_last_mini = input_data[:, :, -1, :, :].cuda() # t frame

            # squeeze the D dimension to C dimension, shape comes to [N, C, H, W]
            input_data_mini = input_data.reshape(input_data.shape[0], -1, input_data.shape[-2], input_data.shape[-1]).cuda()
            output_pred_G = self.G(input_data_mini)
            gtFlow, _ = flow_batch_estimate(self.F, torch.cat([input_last_mini, target_mini], 1), self.normalize.param['train'])
            predFlow, _ = flow_batch_estimate(self.F, torch.cat([input_last_mini, output_pred_G], 1), self.normalize.param['train'])
            frame_psnr_mini = psnr_error(output_pred_G.detach(), target_mini, hat=True)
            flow_psnr_mini = psnr_error(predFlow, gtFlow)
            temp_meter_frame.update(frame_psnr_mini.detach())
            temp_meter_flow.update(flow_psnr_mini.detach())
        self.logger.info(f'&^*_*^& ==> Step:{current_step}/{self.steps.param["max"]} the frame PSNR is {temp_meter_frame.avg:.2f}, the flow PSNR is {temp_meter_flow.avg:.2f}')

    
@ENGINE_REGISTRY.register()
class ANOPREDInference(DefaultInference):
    NAME = ["ANOPRED.INFERENCE"]
    def custom_setup(self):
        if self.kwargs['parallel']:
            self.G = self.data_parallel(self.model['Generator']).load_state_dict(self.save_model['G'])
            self.D = self.data_parallel(self.model['Discriminator']).load_state_dict(self.save_model['D'])
            self.F = self.data_parallel(self.model['FlowNet'])
        else:
            # import ipdb; ipdb.set_trace()
            self.G = self.model['Generator'].cuda()
            self.G.load_state_dict(self.save_model['G'])
            self.D = self.model['Discriminator'].cuda()
            self.D.load_state_dict(self.save_model['D'])
            self.F = self.model['FlowNet'].cuda()
        
        self.test_dataset_keys = self.kwargs['test_dataset_keys']
        self.test_dataset_dict = self.kwargs['test_dataset_dict']
    
    def inference(self):
        for h in self._hooks:
            h.inference()