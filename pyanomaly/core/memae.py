'''
this is the trainer of the 'Memoryizing Normality to detect anomaly: memory-augmented deep Autoencoder for Unsupervised anomaly detection(iccv2019)'
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
from pyanomaly.utils.flow_utils import flow2img
from pyanomaly.core.engine.default_engine import DefaultTrainer, DefaultInference

class Trainer(DefaultTrainer):
    NAME = ["MEMAE.TRAIN"]
    def custom_setup(self):
        if self.kwargs['parallel']:
            self.MemAE = self.data_parallel(self.model['MemAE'])
        else:
            self.MemAE = self.model['MemAE'].cuda()
        
        # get the optimizer
        self.optim_MemAE = self.optimizer['optimizer_memae']

        # get the loss_fucntion
        self.rec_loss = self.loss_function['rec_loss']
        self.mem_loss = self.loss_function['mem_loss']

        # the lr scheduler
        self.lr_memae = self.scheduler_dict['optimizer_memae_scheduler']

        # basic meter
        self.loss_meter_MemAE = AverageMeter(name='loss_memae')
        
        self.test_dataset_keys = self.kwargs['test_dataset_keys']
        self.test_dataset_dict = self.kwargs['test_dataset_dict']

    
    def train(self,current_step):
        # Pytorch [N, C, D, H, W]
        # initialize
        start = time.time()
        self.set_requires_grad(self.MemAE, True)
        self.MemAE.train()
        writer = self.kwargs['writer_dict']['writer']
        global_steps = self.kwargs['writer_dict']['global_steps_{}'.format(self.kwargs['model_type'])]
        
        # get the data
        data, _ = next(self._train_loader_iter)  # the core for dataloader
        self.data_time.update(time.time() - start)
        
        input_data = data.cuda() 
        
        # True Process =================Start===================
        output_rec, att = self.MemAE(input_data)
        loss_rec = self.rec_loss(output_rec, input_data)
        loss_mem = self.mem_loss(att)
        # import ipdb; ipdb.set_trace()
        loss_memae_all = self.loss_lamada['rec_loss'] * loss_rec + self.loss_lamada['mem_loss'] * loss_mem 
        self.optim_MemAE.zero_grad()
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
        
        self.saved_model = {'MemAE':self.MemAE}
        self.saved_optimizer = {'optim_MemAE': self.optim_MemAE}
        self.saved_loss = {'loss_MemAE':self.loss_meter_MemAE.val}
        self.kwargs['writer_dict']['global_steps_{}'.format(self.kwargs['model_type'])] = global_steps
    
    def mini_eval(self, current_step):
        if current_step % self.steps.param['mini_eval']!= 0:
            return
        temp_meter_frame = AverageMeter()
        self.set_requires_grad(self.MemAE, False)
        self.MemAE.eval()
        for data, _ in self.val_dataloader:
            # get the data
            input_data_mini = data.cuda()
            output_rec, _ = self.MemAE(input_data_mini)
            frame_psnr_mini = psnr_error(output_rec.detach(), input_data_mini)
            temp_meter_frame.update(frame_psnr_mini.detach())
        self.logger.info(f'&^*_*^& ==> Step:{current_step}/{self.steps.param["max"]} the frame PSNR is {temp_meter_frame.avg:.3f}')
        # return temp_meter.avg


class Inference(DefaultInference):
    NAME = ["MEMAE.INFERENCE"]
    def custom_setup(self):
        if self.kwargs['parallel']:
            self.MemAE = self.data_parallel(self.model['MemAE']).load_state_dict(self.save_model['MemAE'])
        else:
            self.MemAE = self.model['MemAE'].cuda()
            self.MemAE.load_state_dict(self.save_model['MemAE'])
        
        self.test_dataset_keys = self.kwargs['test_dataset_keys']
        self.test_dataset_dict = self.kwargs['test_dataset_dict']

    
    def inference(self):
        for h in self._hooks:
            h.inference()