'''
this is the core function of the 'AnoPCN: Video Anomaly Detection via Deep Predictive Coding Network'
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

from pyanomaly.core.utils import AverageMeter, flow_batch_estimate, tensorboard_vis_images, make_info_message, ParamSet
from pyanomaly.datatools.evaluate.utils import psnr_error
from pyanomaly.core.engine.default_engine import DefaultTrainer, DefaultInference

from ..engine_registry import ENGINE_REGISTRY

__all__ = ['ANOPCNTrainer', 'ANOPCNInference']
@ENGINE_REGISTRY.register()
class ANOPCNTrainer(DefaultTrainer):
    _NAME = ["ANOPCN.TRAIN"]
    def custom_setup(self):
        # basic things
        if self.kwargs['parallel']:
            self.G = self.data_parallel(self.model['Generator'])
            self.D = self.data_parallel(self.model['Discriminator'])
            self.F = self.data_parallel(self.model['FlowNet'])
        else:
            self.G = self.model['Generator'].cuda()
            self.D = self.model['Discriminator'].cuda()
            self.F = self.model['FlowNet'].cuda()
        
        # get the optimizer
        self.optim_G = self.optimizer['optimizer_g']
        self.optim_D = self.optimizer['optimizer_d']

        # get the loss_fucntion
        self.gan_loss = self.loss_function['gan_loss_mse']
        self.gd_loss = self.loss_function['gradient_loss']
        self.int_loss = self.loss_function['intentsity_loss']
        self.op_loss = self.loss_function['opticalflow_loss_sqrt']

        self.lr_g = self.lr_scheduler_dict['optimizer_g_scheduler']
        self.lr_d = self.lr_scheduler_dict['optimizer_d_scheduler']

        # basic meter
        self.loss_predmeter_G = AverageMeter(name='loss_pred_G')
        self.loss_predmeter_D = AverageMeter(name='loss_pred_D')
        self.loss_refinemeter_G = AverageMeter(name='loss_refine_G')
        self.loss_refinemeter_D = AverageMeter(name='loss_refine_D')

        # others
        self.test_dataset_keys = self.kwargs['test_dataset_keys']
        self.test_dataset_dict = self.kwargs['test_dataset_dict']

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
        
        loss_g_adv = self.gan_loss(self.D(output_predframe_G), True)
        loss_op = self.op_loss(predFlow, gtFlow)
        loss_int = self.int_loss(output_predframe_G, target)
        loss_gd = self.gd_loss(output_predframe_G, target)

        loss_g_all = self.loss_lamada['intentsity_loss'] * loss_int + self.loss_lamada['gradient_loss'] * loss_gd + \
                     self.loss_lamada['opticalflow_loss_sqrt'] * loss_op + self.loss_lamada['gan_loss_mse'] * loss_g_adv
        self.optim_G.zero_grad()
        loss_g_all.backward()
        self.optim_G.step()
        # record
        self.loss_predmeter_G.update(loss_g_all.detach())
        if self.config.TRAIN.adversarial.scheduler.use:
            self.lr_g.step()
        
        #---------update optim_D ---------------
        self.set_requires_grad(self.D, True)
        self.optim_D.zero_grad()
        temp_t = self.D(target)
        temp_g = self.D(output_predframe_G.detach())
        loss_d_1 = self.gan_loss(temp_t, True)
        loss_d_2 = self.gan_loss(temp_g, False)
        loss_d = (loss_d_1 + loss_d_2) * 0.5
        loss_d.backward()
        self.optim_D.step()
        if self.config.TRAIN.adversarial.scheduler.use:
            self.lr_d.step()
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
        self.saved_optimizer = {'optim_G': self.optim_G, 'optim_D': self.optim_D}
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
        
        loss_g_adv = self.gan_loss(self.D(output_refineframe_G), True)
        loss_op = self.op_loss(predFlow, gtFlow)
        loss_int = self.int_loss(output_refineframe_G, target)
        loss_gd = self.gd_loss(output_refineframe_G, target)

        loss_g_all = self.loss_lamada['intentsity_loss'] * loss_int + self.loss_lamada['gradient_loss'] * loss_gd + \
                     self.loss_lamada['opticalflow_loss_sqrt'] * loss_op + self.loss_lamada['gan_loss_mse'] * loss_g_adv
        self.optim_G.zero_grad()
        loss_g_all.backward()
        self.optim_G.step()
        # record
        self.loss_refinemeter_G.update(loss_g_all.detach())
        if self.config.TRAIN.adversarial.scheduler.use:
            self.lr_g.step()
        
        #---------update optim_D ---------------
        self.set_requires_grad(self.D, True)
        self.optim_D.zero_grad()
        temp_t = self.D(target)
        temp_g = self.D(output_refineframe_G.detach())
        loss_d_1 = self.gan_loss(temp_t, True)
        loss_d_2 = self.gan_loss(temp_g, False)
        loss_d = (loss_d_1 + loss_d_2) * 0.5
        loss_d.backward()
        self.optim_D.step()

        if self.config.TRAIN.adversarial.scheduler.use:
            self.lr_d.step()
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
        
        self.saved_model = {'G':self.G, 'D':self.D}
        self.saved_optimizer = {'optim_G': self.optim_G, 'optim_D': self.optim_D}
        self.saved_loss = {'loss_G':self.loss_refinemeter_G.val, 'loss_D':self.loss_refinemeter_D.val}
        self.kwargs['writer_dict']['global_steps_{}'.format(self.kwargs['model_type'])] = global_steps


    def mini_eval(self, current_step):
        if current_step % self.steps.param['mini_eval'] != 0:
            return
        temp_meter = AverageMeter(name='temp')
        self.set_requires_grad(self.F, False)
        self.set_requires_grad(self.D, False)
        self.set_requires_grad(self.G, False)
        self.G.eval()
        self.D.eval()
        self.F.eval()
        for data, _ in self.val_dataloader:
            # get the data
            target_mini = data[:, :, -1, :, :].cuda() # t frame
            input_data_mini = data[:, :, :-1, :, :].cuda() # 0 ~ t-1 frame
            _, output_refineframe_G_mini = self.G(input_data_mini, target_mini)
            vaild_psnr = psnr_error(output_refineframe_G_mini.detach(), target_mini, hat=False)
            temp_meter.update(vaild_psnr.detach())
        self.logger.info(f'&^*_*^& ==> Step:{current_step}/{self.steps.param["max"]} the PSNR is {temp_meter.avg:.3f}')

@ENGINE_REGISTRY.register()
class ANOPCNInference(DefaultInference):
    _NAME = ["ANOPCN.INFERENCE"]
    def __init__(self, *defaults,**kwargs):
        if kwargs['parallel']:
            self.G = self.data_parallel(self.model['Generator']).load_state_dict(self.save_model['G'])
            self.D = self.data_parallel(self.model['Discriminator']).load_state_dict(self.save_model['D'])
            self.F = self.data_parallel(self.model['FlowNet'])
        else:
            self.G = self.model['Generator'].cuda()
            self.G.load_state_dict(self.save_model['G'])
            self.D = self.model['Discriminator'].cuda()
            self.D.load_state_dict(self.save_model['D'])
            self.F = self.model['FlowNet'].cuda()
        

        self.test_dataset_keys = kwargs['test_dataset_keys']
        self.test_dataset_dict = kwargs['test_dataset_dict']

    def inference(self):
        for h in self._hooks:
            h.inference()
    