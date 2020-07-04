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


class Trainer(DefaultTrainer):
    NAME = ["ANOPCN.TRAIN"]
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
        # print('in AnoPredTrainer')
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
        self.gan_loss = loss_function['gan_loss_mse']
        self.gd_loss = loss_function['gradient_loss']
        self.int_loss = loss_function['intentsity_loss']
        self.op_loss = loss_function['opticalflow_loss_sqrt']

        # basic meter
        self.batch_time =  AverageMeter(name='batch_time')
        self.data_time = AverageMeter(name='data_time')
        self.loss_predmeter_G = AverageMeter(name='loss_predmeter_G')
        self.loss_predmeter_D = AverageMeter(name='loss_predmeter_D')
        self.loss_refinemeter_G = AverageMeter(name='loss_refinemeter_G')
        self.loss_refinemeter_D = AverageMeter(name='loss_refinemeter_D')
        # self.psnr = AverageMeter()

        # others
        self.verbose = kwargs['verbose']
        self.accuarcy = 0.0  # to store the accuracy varies from epoch to epoch
        self.config_name = kwargs['config_name']
        self.kwargs = kwargs
        self.result_path = ''
        
        self.normalize = ParamSet(name='normalize', 
                                  train={'use':self.config.ARGUMENT.train.normal.use, 'mean':self.config.ARGUMENT.train.normal.mean, 'std':self.config.ARGUMENT.train.normal.std}, 
                                  val={'use':self.config.ARGUMENT.val.normal.use, 'mean':self.config.ARGUMENT.val.normal.mean, 'std':self.config.ARGUMENT.val.normal.std})

        self.steps = ParamSet(name='steps', log=self.config.TRAIN.log_step, vis=self.config.TRAIN.vis_step, eval=self.config.TRAIN.eval_step, save=self.config.TRAIN.save_step, 
                              max=self.config.TRAIN.max_steps, mini_eval=self.config.TRAIN.mini_eval_step, dynamic_steps=self.config.TRAIN.dynamic_steps)
        
        self.optical = ParamSet(name='optical', size=self.config.DATASET.optical_size, output_format=self.config.DATASET.optical_format)
        # import ipdb; ipdb.set_trace()
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
        self.set_requires_grad(self.trainer.F, False)
        self.set_requires_grad(self.trainer.G, True)
        self.set_requires_grad(self.trainer.D, True)

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
        self.G.train()
        self.D.train()
        self.F.eval()
        # self.set_requires_grad(self.F, False)
        if self.kwargs['parallel']:
            self.set_requires_grad(self.G.module.pcm, True)
            self.set_requires_grad(self.G.module.erm, False)
        else:
            self.set_requires_grad(self.G.pcm, True)
            self.set_requires_grad(self.G.erm, False)
        # self.G.change(True)
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

        loss_g_all = self.loss_lamada['intentsity_loss'] * loss_int + self.loss_lamada['gradient_loss'] * loss_gd + self.loss_lamada['opticalflow_loss_sqrt'] * loss_op + self.loss_lamada['gan_loss_mse'] * loss_g_adv
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
        self.G.train()
        self.D.train()
        self.F.eval()
        # self.set_requires_grad(self.F, False)

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
        
        predFlowEstim = torch.cat([pred_last, output_refineframe_G],1).cuda()
        gtFlowEstim = torch.cat([pred_last, target], 1).cuda()

        gtFlow_vis, gtFlow = flow_batch_estimate(self.F, gtFlowEstim, self.normalize.param['train'], 
                                                 output_format=self.config.DATASET.optical_format, optical_size=self.config.DATASET.optical_size)
        predFlow_vis, predFlow = flow_batch_estimate(self.F, predFlowEstim, self.normalize.param['train'], 
                                                     output_format=self.config.DATASET.optical_format, optical_size=self.config.DATASET.optical_size)
        
        loss_g_adv = self.gan_loss(self.D(output_refineframe_G), True)
        loss_op = self.op_loss(predFlow, gtFlow)
        loss_int = self.int_loss(output_refineframe_G, target)
        loss_gd = self.gd_loss(output_refineframe_G, target)

        loss_g_all = self.loss_lamada['intentsity_loss'] * loss_int + self.loss_lamada['gradient_loss'] * loss_gd + self.loss_lamada['opticalflow_loss_sqrt'] * loss_op + self.loss_lamada['gan_loss_mse'] * loss_g_adv
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
            msg = make_info_message(current_step, self.max_steps, self.kwargs['model_type'], self.batch_time, 
                                    self.config.TRAIN.batch_sizes, self.data_time, [self.loss_refinemeter_G, self.loss_refinemeter_D])
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
        self.G.eval()
        self.D.eval()
        self.set_requires_grad(self.F, False)
        self.set_requires_grad(self.D, False)
        self.set_requires_grad(self.G, False)

        for data, _ in self.val_dataloader:
            # get the data
            target_mini = data[:, :, -1, :, :].cuda() # t frame
            input_data_mini = data[:, :, :-1, :, :].cuda() # 0 ~ t-1 frame
            _, output_refineframe_G_mini = self.G(input_data_mini, target_mini)
            vaild_psnr = psnr_error(output_refineframe_G_mini.detach(), target_mini, hat=True)
            temp_meter.update(vaild_psnr.detach())
        self.logger.info(f'&^*_*^& ==> Step:{current_step}/{self.max_steps} the PSNR is {temp_meter.avg:.3f}')


class Inference(DefaultInference):
    NAME = ["ANOPCN.INFERENCE"]
    def __init__(self, *defaults,**kwargs):
        '''
        Args:
            mode: change the mode of inference, can choose: dataset | image
        '''
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

        self.metric = 0.0
        self.evaluate_function = kwargs['evaluate_function']

    def inference(self):
        for h in self._hooks:
            h.inference()
    