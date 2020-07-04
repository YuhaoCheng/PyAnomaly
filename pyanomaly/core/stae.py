'''
this is the trainer of the 'Spatio-Temporal AutoEncoder for Video Anomaly Detection MM2017'
'''
import sys
sys.path.append('../')
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
    NAME = ["STAE.TRAIN"]
    def custom_setup(self):
        # basic things
        if self.kwargs['parallel']:
            self.STAE = self.data_parallel(self.model['STAE'])
        else:
            self.STAE = self.model['STAE'].cuda()
        
        # get the optimizer
        self.optim_STAE = self.optimizer['optimizer_stae']

        # get the loss_fucntion
        self.rec_loss = self.loss_function['rec_loss']
        self.pred_loss = self.loss_function['weighted_pred_loss']
        
        # the lr scheduler
        self.lr_stae = self.scheduler_dict['optimizer_stae_scheduler']

        # basic meter
        self.loss_meter_STAE = AverageMeter(name='loss_STAE')

        self.test_dataset_keys = self.kwargs['test_dataset_keys']
        self.test_dataset_dict = self.kwargs['test_dataset_dict']

    def train(self,current_step):
        # Pytorch [N, C, D, H, W]
        # initialize
        start = time.time()
        self.set_requires_grad(self.STAE, True)
        self.STAE.train()
        writer = self.kwargs['writer_dict']['writer']
        global_steps = self.kwargs['writer_dict']['global_steps_{}'.format(self.kwargs['model_type'])]
        
        # get the data
        data, _  = next(self._train_loader_iter)  # the core for dataloader
        self.data_time.update(time.time() - start)

        # get the reconstruction and prediction video clip
        time_len = data.shape[2]
        rec_time = time_len // 2
        input_rec = data[:, :, 0:rec_time, :, :].cuda() # 0 ~ t//2 frame 
        input_pred = data[:, :, rec_time:time_len, :, :].cuda() # t//2 ~ t frame

        # True Process =================Start===================
        output_rec,  output_pred = self.STAE(input_rec)
        loss_rec = self.rec_loss(output_rec, input_rec)
        loss_pred = self.pred_loss(output_pred, input_pred)
        # print(f'loss_rec:{loss_rec}')
        # print(f'loss_pred:{loss_pred}')
        loss_stae_all = self.loss_lamada['rec_loss'] * loss_rec + self.loss_lamada['weighted_pred_loss'] * loss_pred 
        self.optim_STAE.zero_grad()
        loss_stae_all.backward()
        self.optim_STAE.step()
        self.loss_meter_STAE.update(loss_stae_all.detach())
        
        if self.config.TRAIN.adversarial.scheduler.use:
            self.lr_stae.step()
        # ======================End==================

        self.batch_time.update(time.time() - start)

        if (current_step % self.steps.param['log'] == 0):
            msg = make_info_message(current_step, self.steps.param['max'], self.kwargs['model_type'], self.batch_time, 
                                    self.config.TRAIN.batch_size, self.data_time, [self.loss_meter_STAE])
            self.logger.info(msg)
        
        writer.add_scalar('Train_loss_STAE', self.loss_meter_STAE.val, global_steps)

        if (current_step % self.steps.param['vis'] == 0):
            vis_objects = OrderedDict({
                'train_output_rec': output_rec.detach(),
                'train_output_pred': output_pred.detach(),
                'train_input_rec':  input_rec.detach(),
                'train_input_pred':  input_pred.detach()
            })
            tensorboard_vis_images(vis_objects, writer, global_steps, self.normalize.param['train'])
        global_steps += 1 
        
        # reset start
        start = time.time()
        
        self.saved_model = {'STAE':self.STAE}
        self.saved_optimizer = {'optim_STAE': self.optim_STAE}
        self.saved_loss = {'loss_STAE':self.loss_meter_STAE}
        self.kwargs['writer_dict']['global_steps_{}'.format(self.kwargs['model_type'])] = global_steps
    
    def mini_eval(self, current_step):
        if current_step % self.steps.param['mini_eval'] != 0:
            return
        temp_meter_rec = AverageMeter()
        # temp_meter_pred = AverageMeter()
        self.set_requires_grad(self.STAE, False)
        self.STAE.eval()
        for data, _ in self.val_dataloader:
            input_mini = data.cuda()
            # Use the model, get the output
            output_rec_mini, output_pred_mini = self.STAE(input_mini)
            rec_psnr_mini = psnr_error(output_rec_mini.detach(), input_mini)
            # pred_psnr_mini = psnr_error(output_pred_mini.detach(), input_pred_mini)
            temp_meter_rec.update(rec_psnr_mini.detach())
            # temp_meter_pred.update(pred_psnr_mini.detach())
        self.logger.info(f'&^*_*^& ==> Step:{current_step}/{self.steps.param["max"]} the REC PSNR is {temp_meter_rec.avg:.3f}')

class Inference(DefaultInference):
    NAME = ["STAE.INFERENCE"]
    def custom_setup(self, *defaults,**kwargs):
        if self.kwargs['parallel']:
            self.STAE = self.data_parallel(self.model['STAE']).load_state_dict(self.save_model['STAE'])
        else:
            # import ipdb; ipdb.set_trace()
            self.STAE = self.model['STAE'].cuda()
            self.STAE.load_state_dict(self.save_model['STAE'])
        
        self.test_dataset_keys = self.kwargs['test_dataset_keys']
        self.test_dataset_dict = self.kwargs['test_dataset_dict']

    def inference(self):
        for h in self._hooks:
            h.inference()