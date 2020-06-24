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

from pyanomaly.core.utils import AverageMeter, flow_batch_estimate, tensorboard_vis_images, vis_optical_flow
from pyanomaly.datatools.evaluate.utils import psnr_error
from pyanomaly.utils.flow_utils import flow2img
from pyanomaly.core.engine.default_engine import DefaultTrainer, DefaultInference

class Trainer(DefaultTrainer):
    NAME = ["AMC.TRAIN"]
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
            self.G = self.data_parallel(model['Generator'])
            self.D = self.data_parallel(model['Discriminator'])
            self.F = self.data_parallel(model['FlowNet'])
        else:
            self.G = model['Generator'].cuda()
            self.D = model['Discriminator'].cuda()
            self.F = model['FlowNet'].cuda()
        
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
        self.gan_loss = loss_function['gan_loss']
        self.gd_loss = loss_function['gradient_loss']
        self.int_loss = loss_function['intentsity_loss']
        self.op_loss = loss_function['opticalflow_loss']

        # basic meter
        self.batch_time =  AverageMeter()
        self.data_time = AverageMeter()
        self.loss_meter_G = AverageMeter()
        self.loss_meter_D = AverageMeter()
        # self.psnr = AverageMeter()

        # others
        self.verbose = kwargs['verbose']
        self.accuarcy = 0.0  # to store the accuracy varies from epoch to epoch
        self.config_name = kwargs['config_name']
        self.kwargs = kwargs
        self.train_normalize = self.config.ARGUMENT.train.normal.use
        self.train_mean = self.config.ARGUMENT.train.normal.mean
        self.train_std = self.config.ARGUMENT.train.normal.std
        self.val_normalize = self.config.ARGUMENT.train.normal.use
        self.val_mean = self.config.ARGUMENT.train.normal.mean
        self.val_std = self.config.ARGUMENT.train.normal.std
        # self.total_steps = len(self.train_dataloader)
        self.result_path = ''
        self.log_step = self.config.TRAIN.log_step # how many the steps, we will show the information
        self.vis_step = self.config.TRAIN.vis_step # how many the steps, we will vis 
        self.eval_step = self.config.TRAIN.eval_step 
        self.save_step = self.config.TRAIN.save_step # save the model whatever the acc of the model
        self.max_steps = self.config.TRAIN.max_steps
        # self.testing_data_folder = self.config.DATASET.test_path
        self.test_dataset_keys = kwargs['test_dataset_keys']
        self.test_dataset_dict = kwargs['test_dataset_dict']

        self.test_dataset_keys_w = kwargs['test_dataset_keys_w']
        self.test_dataset_dict_w = kwargs['test_dataset_dict_w']

        self.evaluate_function = kwargs['evaluate_function']
        # hypyer-parameters of loss 
        self.loss_lamada = kwargs['loss_lamada']

        # the lr scheduler
        # import ipdb; ipdb.set_trace()
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
        self.set_requires_grad(self.F, False)
        writer = self.kwargs['writer_dict']['writer']
        global_steps = self.kwargs['writer_dict']['global_steps_{}'.format(self.kwargs['model_type'])]
        
        # get the data
        data, _  = next(self._train_loader_iter)
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
        flow_gt_vis, flow_gt  = flow_batch_estimate(self.F, gt_flow_esti_tensor, optical_size=self.config.DATASET.optical_size, output_format=self.config.DATASET.optical_format, normalize=self.config.ARGUMENT.train.normal.use, mean=self.config.ARGUMENT.train.normal.mean, std=self.config.ARGUMENT.train.normal.std)
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

        if (current_step % self.log_step == 0):
            msg = 'Step: [{0}/{1}]\t' \
                'Type: {cae_type}\t' \
                'Time: {batch_time.val:.2f}s ({batch_time.avg:.2f}s)\t' \
                'Speed: {speed:.1f} samples/s\t' \
                'Data: {data_time.val:.2f}s ({data_time.avg:.2f}s)\t' \
                'Loss_G: {losses_G.val:.5f} ({losses_G.avg:.3f})\t'   \
                'Loss_D:{losses_D.val:.5f}({losses_D.avg:.3f})'.format(current_step, self.max_steps, cae_type=self.kwargs['model_type'], batch_time=self.batch_time, speed=self.config.TRAIN.batch_size/self.batch_time.val, data_time=self.data_time,losses_G=self.loss_meter_G, losses_D=self.loss_meter_D)
            self.logger.info(msg)
        writer.add_scalar('Train_loss_G', self.loss_meter_G.val, global_steps)
        writer.add_scalar('Train_loss_D', self.loss_meter_D.val, global_steps)
        if (current_step % self.vis_step == 0):
            vis_objects = OrderedDict()
            vis_objects['train_target_flow'] =  flow_gt_vis.detach()
            vis_objects['train_output_flow_G'] = vis_optical_flow(output_flow_G.detach(), output_format=self.config.DATASET.optical_format, output_size=(output_flow_G.shape[-2], output_flow_G.shape[-1]), normalize=self.config.ARGUMENT.train.normal.use, mean=self.config.ARGUMENT.train.normal.mean, std=self.config.ARGUMENT.train.normal.std)
            vis_objects['train_target_frame'] =  target.detach()
            vis_objects['train_output_frame_G'] = output_frame_G.detach()
            # import ipdb; ipdb.set_trace()
            tensorboard_vis_images(vis_objects, writer, global_steps, self.train_normalize, self.train_mean, self.train_std)
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
            gtFlow_vis, gtFlow = flow_batch_estimate(self.F, input_gtFlowEstimTensor, output_format=self.config.DATASET.optical_format, normalize=self.config.ARGUMENT.train.normal.use, optical_size=self.config.DATASET.optical_size,mean=self.config.ARGUMENT.train.normal.mean, std=self.config.ARGUMENT.train.normal.mean)
            
            frame_psnr_mini = psnr_error(output_frame_G_mini.detach(), target_mini)
            flow_psnr_mini = psnr_error(output_flow_G_mini.detach(), gtFlow)
            temp_meter_frame.update(frame_psnr_mini.detach())
            temp_meter_flow.update(flow_psnr_mini.detach())
        self.logger.info(f'&^*_*^& ==> Step:{current_step}/{self.max_steps} the frame PSNR is {temp_meter_frame.avg:.3f}, the flow PSNR is {temp_meter_flow.avg:.3f}')
        # return temp_meter.avg


class Inference(DefaultInference):
    NAME = ["AMC.INFERENCE"]
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
            self.G = model['Generator'].cuda()
            self.G.load_state_dict(save_model['G'])
            self.D = model['Discriminator'].cuda()
            self.D.load_state_dict(save_model['D'])
            self.F = model['FlowNet'].cuda()
        
        # self.load()
        self.F.eval()
        self.set_requires_grad(self.F, False)

        self.verbose = kwargs['verbose']
        self.kwargs = kwargs
        self.config_name = kwargs['config_name']
        self.val_normalize = self.config.ARGUMENT.val.normal.use
        self.val_mean = self.config.ARGUMENT.val.normal.mean
        self.val_std = self.config.ARGUMENT.val.normal.std
        # self.mode = kwargs['mode']

        self.test_dataset_keys = kwargs['test_dataset_keys']
        self.test_dataset_dict = kwargs['test_dataset_dict']

        self.test_dataset_keys_w = kwargs['test_dataset_keys_w']
        self.test_dataset_dict_w = kwargs['test_dataset_dict_w']
        self.metric = 0.0

        self.evaluate_function = kwargs['evaluate_function']

    def inference(self):
        for h in self._hooks:
            h.inference()
    