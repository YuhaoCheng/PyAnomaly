'''
this is the trainer of the 'Anomaly Detection in Video Sequence with Appearance-Motion Correspondence(ICCV2019)'
'''
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
    _NAME = ["AMC.TRAIN"]    
    def custom_setup(self):
        # get model
        if self.kwargs['parallel']:
            self.G = self.data_parallel(self.model['Generator'])
            self.D = self.data_parallel(self.model['Discriminator'])
            self.F = self.data_parallel(self.model['FlowNet'])
        else:
            self.G = self.model['Generator'].cuda()
            self.D = self.model['Discriminator'].cuda()
            self.F = self.model['FlowNet'].cuda()

        # get optimizer
        self.optim_G = self.optimizer['optimizer_g']
        self.optim_D = self.optimizer['optimizer_d']

        # get loss functions
        self.gan_loss = self.loss_function['gan_loss']
        self.gd_loss = self.loss_function['gradient_loss']
        self.int_loss = self.loss_function['intentsity_loss']
        self.op_loss = self.loss_function['opticalflow_loss']

        # get scheculers
        self.lr_g = self.lr_scheduler_dict['optimizer_g_scheduler']
        self.lr_d = self.lr_scheduler_dict['optimizer_d_scheduler']
        
        # create loss meters
        self.loss_meter_G = AverageMeter(name='Loss_G')
        self.loss_meter_D = AverageMeter(name='Loss_D')

        # get test datasets
        self.test_dataset_keys = self.kwargs['test_dataset_keys']
        self.test_dataset_dict = self.kwargs['test_dataset_dict']
        self.test_dataset_keys_w = self.kwargs['test_dataset_keys_w']
        self.test_dataset_dict_w = self.kwargs['test_dataset_dict_w']

        self.optical = ParamSet(name='optical', size=self.config.DATASET.optical_size, output_format=self.config.DATASET.optical_format)
    
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
        loss_g_adv = self.gan_loss(fake_g, True)
        loss_op = self.op_loss(output_flow_G, flow_gt)
        loss_int = self.int_loss(output_frame_G, target)
        loss_gd = self.gd_loss(output_frame_G, target)

        loss_g_all = self.loss_lamada['intentsity_loss'] * loss_int + self.loss_lamada['gradient_loss'] * loss_gd + self.loss_lamada['opticalflow_loss'] * loss_op + self.loss_lamada['gan_loss'] * loss_g_adv
        self.optim_G.zero_grad()
        loss_g_all.backward()
        self.optim_G.step()
        self.loss_meter_G.update(loss_g_all.detach())
        
        if self.config.TRAIN.adversarial.scheduler.use:
            self.lr_g.step()

        #---------update optim_D ---------------
        self.set_requires_grad(self.D, True)
        self.optim_D.zero_grad()
        # import ipdb; ipdb.set_trace()
        real_d = self.D(torch.cat([target, flow_gt],dim=1))
        fake_d = self.D(torch.cat([target, output_flow_G.detach()], dim=1))
        loss_d_1 = self.gan_loss(real_d, True)
        loss_d_2 = self.gan_loss(fake_d, False)
        loss_d = (loss_d_1  + loss_d_2) * 0.5 
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
        
        self.saved_model = {'G':self.G, 'D':self.D}
        self.saved_optimizer = {'optim_G': self.optim_G, 'optim_D': self.optim_D}
        self.saved_loss = {'loss_G':self.loss_meter_G.val, 'loss_D':self.loss_meter_D.val}
        self.kwargs['writer_dict']['global_steps_{}'.format(self.kwargs['model_type'])] = global_steps
    
    def mini_eval(self, current_step):
        if current_step % self.steps.param['mini_eval'] != 0:
            return
        temp_meter_frame = AverageMeter()
        temp_meter_flow = AverageMeter()
        self.set_requires_grad(self.F, False)
        self.set_requires_grad(self.D, False)
        self.set_requires_grad(self.G, False)
        self.G.eval()
        self.D.eval()
        self.F.eval()
        for data , _ in self.val_dataloader:
            # get the data
            target_mini = data[:, :, 1, :, :]
            input_data_mini = data[:, :, 0, :, :]
            # squeeze the dimension
            target_mini = target_mini.view(target_mini.shape[0],-1,target_mini.shape[-2], target_mini.shape[-1]).cuda()
            input_data_mini = input_data_mini.view(input_data_mini.shape[0],-1,input_data_mini.shape[-2], input_data_mini.shape[-1]).cuda()
            
            # Use the model, get the output
            output_flow_G_mini, output_frame_G_mini = self.G(input_data_mini)
            input_gtFlowEstimTensor = torch.cat([input_data_mini, target_mini], 1)
            gtFlow_vis, gtFlow = flow_batch_estimate(self.F, input_gtFlowEstimTensor, self.trainer.normalize.param['val'],
                                                    output_format=self.config.DATASET.optical_format, optical_size=self.config.DATASET.optical_size)
            
            frame_psnr_mini = psnr_error(output_frame_G_mini.detach(), target_mini)
            flow_psnr_mini = psnr_error(output_flow_G_mini.detach(), gtFlow)
            temp_meter_frame.update(frame_psnr_mini.detach())
            temp_meter_flow.update(flow_psnr_mini.detach())
        self.logger.info(f'&^*_*^& ==> Step:{current_step}/{self.steps.param["max"]} the frame PSNR is {temp_meter_frame.avg:.3f}, the flow PSNR is {temp_meter_flow.avg:.3f}')
        # return temp_meter.avg


@ENGINE_REGISTRY.register()
class AMCInference(DefaultInference):
    _NAME = ["AMC.INFERENCE"]
    def custom_setup(self):
        if self.kwargs['parallel']:
            self.G = self.data_parallel(self.model['Generator']).load_state_dict(self.save_model['G'])
            self.D = self.data_parallel(self.model['Discriminator']).load_state_dict(self.save_model['D'])
            self.F = self.data_parallel(self.model['FlowNet'])
        else:
            self.G = self.model['Generator'].cuda()
            self.G.load_state_dict(self.save_model['G'])
            self.D = self.model['Discriminator'].cuda()
            self.D.load_state_dict(self.save_model['D'])
            self.F = self.model['FlowNet'].cuda()
        
        self.test_dataset_keys = self.kwargs['test_dataset_keys']
        self.test_dataset_dict = self.kwargs['test_dataset_dict']

        self.test_dataset_keys_w = self.kwargs['test_dataset_keys_w']
        self.test_dataset_dict_w = self.kwargs['test_dataset_dict_w']

    def inference(self):
        for h in self._hooks:
            h.inference()
    