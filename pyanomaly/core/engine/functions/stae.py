"""
@author:  Yuhao Cheng
@contact: yuhao.cheng[at]outlook.com
"""
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
from ..abstract.default_engine import DefaultTrainer, DefaultInference
from ..engine_registry import ENGINE_REGISTRY

__all__ = ['STAETrainer', 'STAEInference']

@ENGINE_REGISTRY.register()
class STAETrainer(DefaultTrainer):
    NAME = ["STAE.TRAIN"]
    def custom_setup(self):

        # for item_key in self.model.keys():
        #     attr_name = str(item_key)
        #     if self.kwargs['parallel']:
        #         temp_model = self.data_parallel(self.model[item_key])
        #     else:
        #         temp_model = self.model[item_key].cuda()
        #     self.__setattr__(attr_name, temp_model)
        
        # # get the optimizer
        # for item_key in self.optimizer.keys():
        #     attr_name = str(item_key)
        #     # get the optimizer
        #     self.__setattr__(attr_name, self.optimizer[item_key])
        #     # get the lr scheduler
        #     self.__setattr__(f'{attr_name}_scheduler', self.lr_scheduler_dict[f'{attr_name}_scheduler'])


        # import ipdb; ipdb.set_trace()

        # get the loss_fucntion
        # self.rec_loss = self.loss_function['RecLoss']
        # self.pred_loss = self.loss_function['WeightedPredLoss']
        
        

        # basic meter
        self.loss_meter_STAE = AverageMeter(name='loss_STAE')


    def train(self,current_step):
        # Pytorch [N, C, D, H, W]
        # initialize
        start = time.time()
        self.set_requires_grad(self.STAE, True)
        self.STAE.train()
        writer = self.kwargs['writer_dict']['writer']
        global_steps = self.kwargs['writer_dict']['global_steps_{}'.format(self.kwargs['model_type'])]
        
        # get the data
        data, anno, meta  = next(self._train_loader_iter)  # the core for dataloader
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

        # loss_stae_all = self.loss_lamada['rec_loss'] * loss_rec + self.loss_lamada['weighted_pred_loss'] * loss_pred 
        loss_stae_all = self.loss_lamada['RecLoss'] * loss_rec + self.loss_lamada['WeightedPredLoss'] * loss_pred 
        # self.optim_STAE.zero_grad()
        self.optimizer_STAE.zero_grad()
        loss_stae_all.backward()
        # self.optim_STAE.step()
        self.optimizer_STAE.step()
        self.loss_meter_STAE.update(loss_stae_all.detach())
        
        if self.config.TRAIN.general.scheduler.use:
            # self.lr_stae.step()
            self.optimizer_STAE_scheduler.step()
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
        
        # self.saved_model = {'STAE':self.STAE}
        self.saved_model['STAE'] = self.STAE
        # self.saved_optimizer = {'optim_STAE': self.optim_STAE}
        # self.saved_optimizer = {'optim_STAE': self.optimizer_STAE}
        self.saved_optimizer['optimizer_STAE'] = self.optimizer_STAE
        # self.saved_loss = {'loss_STAE':self.loss_meter_STAE}
        self.saved_loss['loss_STAE'] = self.loss_meter_STAE
        self.kwargs['writer_dict']['global_steps_{}'.format(self.kwargs['model_type'])] = global_steps
    
@ENGINE_REGISTRY.register()
class STAEInference(DefaultInference):
    NAME = ["STAE.INFERENCE"]
    def custom_setup(self, *defaults,**kwargs):
        # if self.kwargs['parallel']:
        #     self.STAE = self.data_parallel(self.model['STAE']).load_state_dict(self.save_model['STAE'])
        # else:
        #     # import ipdb; ipdb.set_trace()
        #     self.STAE = self.model['STAE'].cuda()
        #     self.STAE.load_state_dict(self.save_model['STAE'])
        
        # self.test_dataset_keys = self.kwargs['test_dataset_keys']
        # self.test_dataset_dict = self.kwargs['test_dataset_dict']
        pass

    def inference(self):
        for h in self._hooks:
            h.inference()