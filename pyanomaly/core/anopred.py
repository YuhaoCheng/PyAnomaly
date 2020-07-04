'''
this is the trainer of the 'Future Frame Prediction for Anomaly Detection - A New Baseline CVPR2018'
'''
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
from pyanomaly.core.engine.default_engine import DefaultTrainer, DefaultInference

class Trainer(DefaultTrainer):
    NAME = ["ANOPRED.TRAIN"]
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
        # print('in AnoPredTrainer')
        # logger & config
        self._hooks = []
        self._register_hooks(kwargs['hooks'])
        self.logger = defaults[5]
        self.config = defaults[6]

        model = defaults[0]
        # basic things
        if kwargs['parallel']:
            self.G = self.data_parallel(model['Generator'])
            self.D = self.data_parallel(model['Discriminator'])
            self.F = self.data_parallel(model['FlowNet'])
        else:
            self.G = model['Generator'].cuda()
            self.D = model['Discriminator'].cuda()
            self.F = model['FlowNet'].cuda() # lite flownet
        
        
        if kwargs['pretrain']:
            self.load_pretrain()

        self.train_dataloader = defaults[1]
        self._train_loader_iter = iter(self.train_dataloader)

        self.val_dataloader = defaults[2]
        self._val_loader_iter = iter(self.val_dataloader)

        # get the optimizer
        optimizer = defaults[3]
        self.optim_G = optimizer['optimizer_g']
        self.optim_D = optimizer['optimizer_d']

        # get the loss_fucntion
        loss_function = defaults[4]
        self.gan_loss = loss_function['gan_loss_mse']
        self.gd_loss = loss_function['gradient_loss']
        self.int_loss = loss_function['intentsity_loss']
        self.op_loss = loss_function['opticalflow_loss']

        # basic meter
        self.batch_time =  AverageMeter(name='batch_time')
        self.data_time = AverageMeter(name='data_time')
        self.loss_meter_G = AverageMeter(name='loss_G')
        self.loss_meter_D = AverageMeter(name='loss_D')
        # self.psnr = AverageMeter()

        # others
        self.verbose = kwargs['verbose']
        self.accuarcy = 0.0  # to store the accuracy varies from epoch to epoch
        self.config_name = kwargs['config_name']
        self.result_path = ''
        self.kwargs = kwargs

        self.normalize = ParamSet(name='normalize', 
                                  train={'use':self.config.ARGUMENT.train.normal.use, 'mean':self.config.ARGUMENT.train.normal.mean, 'std':self.config.ARGUMENT.train.normal.std}, 
                                  val={'use':self.config.ARGUMENT.val.normal.use, 'mean':self.config.ARGUMENT.val.normal.mean, 'std':self.config.ARGUMENT.val.normal.std})
        self.steps = ParamSet(name='steps', log=self.config.TRAIN.log_step, vis=self.config.TRAIN.vis_step, eval=self.config.TRAIN.eval_step, save=self.config.TRAIN.save_step, 
                              max=self.config.TRAIN.max_steps, mini_eval=self.config.TRAIN.mini_eval_step, dynamic_steps=self.config.TRAIN.dynamic_steps)
        self.optical = ParamSet(name='optical', size=self.config.DATASET.optical_size, output_format=self.config.DATASET.optical_format)

        # self.testing_data_folder = self.config.DATASET.test_path
        self.test_dataset_keys = kwargs['test_dataset_keys']
        self.test_dataset_dict = kwargs['test_dataset_dict']

        self.evaluate_function = kwargs['evaluate_function']
        
        # hypyer-parameters of loss 
        self.loss_lamada = kwargs['loss_lamada']

        # the lr scheduler
        scheduler_dict = kwargs['lr_scheduler_dict']
        self.lr_g = scheduler_dict['optimizer_g_scheduler']
        self.lr_d = scheduler_dict['optimizer_d_scheduler']

        if self.config.RESUME.flag:
            self.resume()
        
        if self.config.FINETUNE.flag:
            self.fine_tune()
    

    def train(self,current_step):
        # Pytorch [N, C, D, H, W]
        # initialize
        start = time.time()
        self.G.train()
        self.D.train()
        self.F.eval()
        self.set_requires_grad(self.F, False)
        writer = self.kwargs['writer_dict']['writer']
        global_steps = self.kwargs['writer_dict']['global_steps_{}'.format(self.kwargs['model_type'])]

        # get the data
        data, _ = next(self._train_loader_iter) 
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
        self.G.eval()
        self.D.eval()
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

    

class Inference(DefaultInference):
    NAME = ["ANOPRED.INFERENCE"]
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
            self.G = self.data_parallel(model['Generator']).load_state_dict(save_model['G'])
            self.D = self.data_parallel(model['Discriminator']).load_state_dict(save_model['D'])
            self.F = self.data_parallel(model['FlowNet'])
        else:
            # import ipdb; ipdb.set_trace()
            self.G = model['Generator'].cuda()
            self.G.load_state_dict(save_model['G'])
            self.D = model['Discriminator'].cuda()
            self.D.load_state_dict(save_model['D'])
            self.F = model['FlowNet'].cuda()
        
        # self.load()

        self.verbose = kwargs['verbose']
        self.kwargs = kwargs
        self.config_name = kwargs['config_name']
        
        self.normalize = ParamSet(name='normalize', 
                                  train={'use':self.config.ARGUMENT.train.normal.use, 'mean':self.config.ARGUMENT.train.normal.mean, 'std':self.config.ARGUMENT.train.normal.std}, 
                                  val={'use':self.config.ARGUMENT.val.normal.use, 'mean':self.config.ARGUMENT.val.normal.mean, 'std':self.config.ARGUMENT.val.normal.std})
        
        self.test_dataset_keys = kwargs['test_dataset_keys']
        self.test_dataset_dict = kwargs['test_dataset_dict']

        self.metric = 0.0
        self.evaluate_function = kwargs['evaluate_function']
    
    def inference(self):
        for h in self._hooks:
            h.inference()