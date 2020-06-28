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

from pyanomaly.core.utils import AverageMeter, flow_batch_estimate, tensorboard_vis_images
from pyanomaly.datatools.evaluate.utils import psnr_error
from pyanomaly.utils.flow_utils import flow2img
from pyanomaly.core.engine.default_engine import DefaultTrainer, DefaultInference

class Trainer(DefaultTrainer):
    NAME = ["STAE.TRAIN"]
    def __init__(self, *defaults, **kwargs):
        '''
        Args:
            defaults(tuple): the default will have:
                0->model:{'Generator':net_g, 'Driscriminator':net_d, 'FlowNet':net_flow}
                1->train_dataloader: the dataloader   
                2->val_dataloader: the dataloader     
                3->optimizer:{'optimizer_g':op_g, 'optimizer_d'}
                4->loss_function: {'g_adverserial_loss':.., 'd_adverserial_loss':..., 'gradient_loss':.., 'opticalflow_loss':.., 'intentsity_loss':.. }
                5->logger: the logger of the whole training process
                6->config: the config object of the whole process

            kwargs(dict): the default will have:
                verbose(str):
                parallel(bool): True-> data parallel
                pertrain(bool): True-> use the pretarin model
                extra param:
                    test_dataset_keys: the dataset keys of each video
                    test_dataset_dict: the dataset dict of whole test videos
        '''
        self._hooks = []
        self._eval_hooks = []
        self._register_hooks(kwargs['hooks'])
        # logger & config
        self.logger = defaults[5]
        self.config = defaults[6]

        model = defaults[0]
        # basic things
        if kwargs['parallel']:
            self.STAE = self.data_parallel(model['STAE'])
        else:
            self.STAE = model['STAE'].cuda()
        
        if kwargs['pretrain']:
            self.load_pretrain()

        self.train_dataloader = defaults[1]
        self._train_loader_iter = iter(self.train_dataloader)

        self.val_dataloader = defaults[2]
        self._val_loader_iter = iter(self.val_dataloader)

        # get the optimizer
        optimizer = defaults[3]
        self.optim_STAE = optimizer['optimizer_stae']

        # get the loss_fucntion
        loss_function = defaults[4]
        # self.rec_loss = loss_function['rec_loss']
        self.rec_loss = loss_function['rec_loss']
        # self.pred_loss = loss_function['pred_loss']
        self.pred_loss = loss_function['weighted_pred_loss']

        # basic meter
        self.batch_time =  AverageMeter()
        self.data_time = AverageMeter()
        self.loss_meter_STAE = AverageMeter()

        # others
        self.verbose = kwargs['verbose']
        self.accuarcy = 0.0  # to store the accuracy varies from epoch to epoch
        self.config_name = kwargs['config_name']
        self.kwargs = kwargs
        self.train_normalize = self.config.ARGUMENT.train.normal.use
        self.train_mean = self.config.ARGUMENT.train.normal.mean
        self.train_std = self.config.ARGUMENT.train.normal.std
        self.val_normalize = self.config.ARGUMENT.val.normal.use
        self.val_mean = self.config.ARGUMENT.val.normal.mean
        self.val_std = self.config.ARGUMENT.val.normal.std
        # self.total_steps = len(self.train_dataloader)
        self.result_path = ''
        self.log_step = self.config.TRAIN.log_step # how many the steps, we will show the information
        self.vis_step = self.config.TRAIN.vis_step # how many the steps, we will vis
        self.eval_step = self.config.TRAIN.eval_step 
        self.save_step = self.config.TRAIN.save_step # save the model whatever the acc of the model
        self.max_steps = self.config.TRAIN.max_steps

        self.test_dataset_keys = kwargs['test_dataset_keys']
        self.test_dataset_dict = kwargs['test_dataset_dict']

        self.evaluate_function = kwargs['evaluate_function']
        # hypyer-parameters of loss 
        self.loss_lamada = kwargs['loss_lamada']

        # the lr scheduler
        scheduler_dict = kwargs['lr_scheduler_dict']
        self.lr_stae = scheduler_dict['optimizer_stae_scheduler']

        if self.config.RESUME.flag:
            self.resume()
        
        if self.config.FINETUNE.flag:
            self.fine_tune()
        
    
    def train(self,current_step):
        # Pytorch [N, C, D, H, W]
        
        # initialize
        start = time.time()
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

        loss_stae_all = self.loss_lamada['rec_loss'] * loss_rec + self.loss_lamada['weighted_pred_loss'] * loss_pred 
        self.optim_STAE.zero_grad()
        loss_stae_all.backward()
        self.optim_STAE.step()
        self.loss_meter_STAE.update(loss_stae_all.detach())
        
        if self.config.TRAIN.adversarial.scheduler.use:
            self.lr_stae.step()
        # ======================End==================

        self.batch_time.update(time.time() - start)

        if (current_step % self.log_step == 0):
            msg = 'Step: [{0}/{1}]\t' \
                'Type: {cae_type}\t' \
                'Time: {batch_time.val:.2f}s ({batch_time.avg:.2f}s)\t' \
                'Speed: {speed:.1f} samples/s\t' \
                'Data: {data_time.val:.2f}s ({data_time.avg:.2f}s)\t' \
                'Loss_STAE: {loss.val:.5f} ({loss.avg:.5f})'.format(current_step, self.max_steps, cae_type=self.kwargs['model_type'], batch_time=self.batch_time, speed=self.config.TRAIN.batch_size/self.batch_time.val, data_time=self.data_time,loss=self.loss_meter_STAE)
            self.logger.info(msg)
        writer.add_scalar('Train_loss_STAE', self.loss_meter_STAE.val, global_steps)
        if (current_step % self.vis_step == 0):
            vis_objects = OrderedDict()
            vis_objects['train_output_rec'] = output_rec.detach()
            vis_objects['train_output_pred'] = output_pred.detach()
            vis_objects['train_input_rec'] =  input_rec.detach()
            vis_objects['train_input_pred'] =  input_pred.detach()
            tensorboard_vis_images(vis_objects, writer, global_steps, self.train_normalize, self.train_mean, self.train_std)
        global_steps += 1 
        
        # reset start
        start = time.time()
        
        self.saved_model = {'STAE':self.STAE}
        self.saved_optimizer = {'optim_STAE': self.optim_STAE}
        self.saved_loss = {'loss_STAE':self.loss_meter_STAE}
        self.kwargs['writer_dict']['global_steps_{}'.format(self.kwargs['model_type'])] = global_steps
    
    def mini_eval(self, current_step):
        if current_step % self.config.TRAIN.mini_eval_step != 0:
            return
        temp_meter_rec = AverageMeter()
        # temp_meter_pred = AverageMeter()
        self.STAE.eval()
        for data, _ in self.val_dataloader:
            # get the reconstruction and prediction video clip
            # time_len = data.shape[2]
            # rec_time = time_len // 2
            # inupt_rec_mini = data[:, :, 0:rec_time, :, :].cuda() # 0 ~ t//2 frame 
            # input_pred_mini = data[:, :, rec_time:time_len, :, :].cuda() # t//2 ~ t frame 
            input_mini = data.cuda()
            # Use the model, get the output
            output_rec_mini, output_pred_mini = self.STAE(input_mini)
            rec_psnr_mini = psnr_error(output_rec_mini.detach(), input_mini)
            # pred_psnr_mini = psnr_error(output_pred_mini.detach(), input_pred_mini)
            temp_meter_rec.update(rec_psnr_mini.detach())
            # temp_meter_pred.update(pred_psnr_mini.detach())
        self.logger.info(f'&^*_*^& ==> Step:{current_step}/{self.max_steps} the REC PSNR is {temp_meter_rec.avg:.3f}')
        # return temp_meter.avg


class Inference(DefaultInference):
    NAME = ["MEMAE.INFERENCE"]
    def __init__(self, *defaults,**kwargs):
        '''
         Args:
            defaults(tuple): the default will have:
                0->model: the model of the experiment
                1->model_path: the path of the model path
                2->val_dataloader: the dataloader to inference
                3->logger: the logger of the whole process
                4->config: the config object of the whole process
            kwargs(dict): the default will have:
                verbose(str):
                parallel(bool): True-> data parallel
                pertrain(bool): True-> use the pretarin model
                mode(str): 'dataset' -> the data will use the dataloder to pass in(dicard, becasue we will use the dataset to get all I need)
        '''
        self._hooks = []
        self._register_hooks(kwargs['hooks'])
        self.logger = defaults[3]
        self.config = defaults[4]
        self.model_path = defaults[1]

        save_model = torch.load(self.model_path)
        
        model = defaults[0]
        if kwargs['parallel']:
            self.G = self.data_parallel(model['STAE']).load_state_dict(save_model['STAE'])
        else:
            # import ipdb; ipdb.set_trace()
            self.G = model['STAE'].cuda()
            self.D.load_state_dict(save_model['STAE'])
        
        # self.load()

        self.verbose = kwargs['verbose']
        self.kwargs = kwargs
        self.config_name = kwargs['config_name']
        self.normalize = self.config.ARGUMENT.val.normal.use
        self.mean = self.config.ARGUMENT.val.normal.mean
        self.std = self.config.ARGUMENT.val.normal.std
        # self.mode = kwargs['mode']

        self.test_dataset_keys = kwargs['test_dataset_keys']
        self.test_dataset_dict = kwargs['test_dataset_dict']

        self.metric = 0.0

        self.evaluate_function = kwargs['evaluate_function']
    
    def inference(self):
        for h in self._hooks:
            h.inference()