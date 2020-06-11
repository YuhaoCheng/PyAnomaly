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

from lib.core.utils import AverageMeter, flow_batch_estimate, training_vis_images
from lib.datatools.evaluate.utils import psnr_error
from lib.utils.flow_utils import flow2img
from lib.core.engine.default_engine import DefaultTrainer, DefaultInference

class Trainer(DefaultTrainer):
    NAME = ["MEMAE.TRAIN"]
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
            self.MemAE = self.data_parallel(model['MemAE'])
        else:
            self.MemAE = model['MemAE'].cuda()
        
        if kwargs['pretrain']:
            self.load_pretrain()

        self.train_dataloader = defaults[1]
        self._train_loader_iter = iter(self.train_dataloader)

        self.val_dataloader = defaults[2]
        self._val_loader_iter = iter(self.val_dataloader)

        # get the optimizer
        optimizer = defaults[3]
        self.optim_MemAE = optimizer['optimizer_memae']

        # get the loss_fucntion
        loss_function = defaults[4]
        self.rec_loss = loss_function['rec_loss']
        self.mem_loss = loss_function['mem_loss']

        # basic meter
        self.batch_time =  AverageMeter()
        self.data_time = AverageMeter()
        self.loss_meter_MemAE = AverageMeter()

        # others
        self.verbose = kwargs['verbose']
        self.accuarcy = 0.0  # to store the accuracy varies from epoch to epoch
        self.config_name = kwargs['config_name']
        self.kwargs = kwargs
        # self.total_steps = len(self.train_dataloader)
        self.result_path = ''
        self.log_step = self.config.TRAIN.log_step # how many the steps, we will show the information
        self.vis_step = self.config.TRAIN.vis_step # how many the steps, we will show the information
        self.eval_step = self.config.TRAIN.eval_step 
        self.save_step = self.config.TRAIN.save_step # save the model whatever the acc of the model
        self.max_steps = self.config.TRAIN.max_steps


        self.evaluate_function = kwargs['evaluate_function']
        # hypyer-parameters of loss 
        self.loss_lamada = kwargs['loss_lamada']

        # the lr scheduler
        scheduler_dict = kwargs['lr_scheduler_dict']
        self.lr_memae = scheduler_dict['optimizer_memae_scheduler']

        if self.config.RESUME.flag:
            self.resume()
        
        if self.config.FINETUNE.flag:
            self.fine_tune()
        
    
    def train(self,current_step):
        start = time.time()
        self.MemAE.train()
        writer = self.kwargs['writer_dict']['writer']
        global_steps = self.kwargs['writer_dict']['global_steps_{}'.format(self.kwargs['model_type'])]
        
        # for step, data in enumerate(self.train_dataloader):
        data  = next(self._train_loader_iter)  # the core for dataloader
        self.data_time.update(time.time() - start)
        # time_len = data.shape[2]
        input = data.cuda() # t+1 frame 
        # input = data[:, :-1, ] # 0~t frame
        # input_last = input[:,-1,].cuda() # t frame
        # input = input.view(input.shape[0], -1, input.shape[-2], input.shape[-1]).cuda() # 0~t frame
        # input = input.reshape(input.shape[0], -1, input,input.shape[-2], input.shape[-1]).cuda() # 0~t frame
        # True Process =================Start===================
        output_rec, att = self.MemAE(input)
        loss_rec = self.rec_loss(output_rec, input)
        loss_mem = self.mem_loss(att)
        print(f'rec:{loss_rec}, loss_mem:{loss_mem}')
        # loss_pred = self.pred_loss(output_pred, pred)

        loss_memae_all = self.loss_lamada['rec_loss'] * loss_rec + self.loss_lamada['mem_loss'] * loss_mem 
        self.optim_MemAE.zero_grad()
        loss_memae_all.backward()
        self.optim_MemAE.step()
        self.loss_meter_MemAE.update(loss_memae_all.detach())
        
        if self.config.TRAIN.adversarial.scheduler.use:
            self.lr_memae.step()
        # ======================End==================

        self.batch_time.update(time.time() - start)

        if (current_step % self.log_step == 0):
            msg = 'Step: [{0}/{1}]\t' \
                'Type: {cae_type}\t' \
                'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                'Speed: {speed:.1f} samples/s\t' \
                'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                'Loss: {losses.val:.5f} ({losses.avg:.5f})'.format(current_step, self.max_steps, cae_type=self.kwargs['model_type'], batch_time=self.batch_time, speed=self.config.TRAIN.batch_size/self.batch_time.val, data_time=self.data_time,losses=self.loss_meter_MemAE)
            self.logger.info(msg)
        writer.add_scalar('Train_loss_MemAE', self.loss_meter_MemAE.val, global_steps)
        if (current_step % self.vis_step == 0):
            vis_objects = OrderedDict()
            vis_objects['train_MemAE_output'] = output_rec.detach()
            vis_objects['train_target_frame'] =  input.detach()
            training_vis_images(vis_objects, writer, global_steps)
        global_steps += 1 
        
        # reset start
        start = time.time()
        
        self.saved_model = {'MemAE':self.MemAE}
        self.saved_optimizer = {'optim_MemAE': self.optim_MemAE}
        self.saved_loss = {'loss_MemAE':self.loss_meter_MemAE.val}
        self.kwargs['writer_dict']['global_steps_{}'.format(self.kwargs['model_type'])] = global_steps
    
    def mini_eval(self, current_step):
        if current_step % self.config.TRAIN.mini_eval_step != 0:
            return
        temp_meter_frame = AverageMeter()
        self.MemAE.eval()
        for data in self.val_dataloader:
            # vaild_target = data[:,-1,].cuda()
            vaild_input = data.cuda()
            # vaild_input_last = vaild_input[:,-1].cuda()
            # vaild_input = vaild_input.reshape(data.shape[0],-1,data.shape[-2], data.shape[-1]).cuda()
            vaild_input = vaild_input.cuda()
            vaild_output_frame, _ = self.MemAE(vaild_input)
            # gt_flow_esti_tensor = torch.cat([vaild_input_last, vaild_target], 1)
            # flow_gt, _ = flow_batch_estimate(self.F, gt_flow_esti_tensor)
            # import ipdb; ipdb.set_trace()
            # vaild_psnr = psnr_error(vaild_output[1].detach(), vaild_target)
            vaild_frame_psnr = psnr_error(vaild_output_frame.detach(), vaild_input)
            # vaild_flow_psnr = psnr_error(vaild_output_flow.detach(), flow_gt)
            temp_meter_frame.update(vaild_frame_psnr.detach())
        self.logger.info(f'&^*_*^& ==> Step:{current_step}/{self.max_steps} the frame PSNR is {temp_meter_frame.avg:.3f}')
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
            self.G = self.data_parallel(model['Generator']).load_state_dict(save_model['G'])
            self.D = self.data_parallel(model['Discriminator']).load_state_dict(save_model['D'])
            # self.G = model['Generator'].to(torch.device('cuda:0'))
            # self.D = model['Discriminator'].to(torch.device('cuda:1'))
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
        # self.mode = kwargs['mode']

        self.test_dataset_keys = kwargs['test_dataset_keys']
        self.test_dataset_dict = kwargs['test_dataset_dict']

        self.metric = 0.0
    
    def inference(self):
        for h in self._hooks:
            h.inference()