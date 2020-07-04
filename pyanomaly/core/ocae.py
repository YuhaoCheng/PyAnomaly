'''
this is the trainer of the 'Future Frame Prediction for Anomaly Detection - A New Baseline CVPR2018'
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
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as tf

from pyanomaly.core.engine.default_engine import DefaultTrainer, DefaultInference
from pyanomaly.core.utils import AverageMeter, multi_obj_grid_crop, frame_gradient, get_batch_dets, tensorboard_vis_images, ParamSet, make_info_message
from pyanomaly.datatools.evaluate.utils import psnr_error

try:
    from sklearn.externals import joblib
except:
    import joblib

class Trainer(DefaultTrainer):
    NAME = ["OCAE.TRAIN"]
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
        self._register_hooks(kwargs['hooks'])
        # logger & config
        self.logger = defaults[5]
        self.config = defaults[6]

        model = defaults[0]
        # basic things
        if kwargs['parallel']:
            self.A = self.data_parallel(model['A'])
            self.B = self.data_parallel(model['B'])
            self.C = self.data_parallel(model['C'])
            self.Detector = self.data_parallel(model['Detector'])
        else:
            self.A = model['A'].cuda()
            self.B = model['B'].cuda()
            self.C = model['C'].cuda()
            self.Detector = model['Detector'].cuda()
        
        self.ovr_model = model['OVR']
        
        if kwargs['pretrain']:
            self.load_pretrain()

        self.train_dataloader = defaults[1]
        self._train_loader_iter = iter(self.train_dataloader)

        self.val_dataloader = defaults[2]
        self._val_loader_iter = iter(self.val_dataloader)

        # get the optimizer
        optimizer = defaults[3]
        self.optim_ABC = optimizer['optimizer_abc']

        # get the loss_fucntion
        loss_function = defaults[4]
        self.a_loss = loss_function['A_loss']
        self.b_loss = loss_function['B_loss']
        self.c_loss = loss_function['C_loss']

        # basic meter
        self.batch_time =  AverageMeter()
        self.data_time = AverageMeter()
        self.loss_meter_ABC = AverageMeter(name='loss_ABC')
        self.psnr = AverageMeter()

        # others
        self.verbose = kwargs['verbose']
        self.accuarcy = 0.0  # to store the accuracy varies from epoch to epoch
        self.config_name = kwargs['config_name']
        self.kwargs = kwargs
        self.ovr_model_path = os.path.join(self.config.TRAIN.model_output, f'ocae_cfg@{self.config_name}#{self.verbose}.npy')

        self.normalize = ParamSet(name='normalize', 
                                  train={'use':self.config.ARGUMENT.train.normal.use, 'mean':self.config.ARGUMENT.train.normal.mean, 'std':self.config.ARGUMENT.train.normal.std}, 
                                  val={'use':self.config.ARGUMENT.val.normal.use, 'mean':self.config.ARGUMENT.val.normal.mean, 'std':self.config.ARGUMENT.val.normal.std})
        # self.total_steps = len(self.train_dataloader)
        self.result_path = ''
        self.steps = ParamSet(name='steps', log=self.config.TRAIN.log_step, vis=self.config.TRAIN.vis_step, eval=self.config.TRAIN.eval_step, save=self.config.TRAIN.save_step, 
                              max=self.config.TRAIN.max_steps, mini_eval=self.config.TRAIN.mini_eval_step, dynamic_steps=self.config.TRAIN.dynamic_steps)

        # self.testing_data_folder = self.config.DATASET.test_path
        self.test_dataset_keys = kwargs['test_dataset_keys']
        self.test_dataset_dict = kwargs['test_dataset_dict']

        self.cluster_dataset_keys = kwargs['cluster_dataset_keys']
        self.cluster_dataset_dict = kwargs['cluster_dataset_dict']

        self.evaluate_function = kwargs['evaluate_function']
        
        # hypyer-parameters of loss 
        self.loss_lamada = kwargs['loss_lamada']

        # the lr scheduler
        lr_scheduler_dict = kwargs['lr_scheduler_dict']
        self.lr_abc = lr_scheduler_dict['optimizer_abc_scheduler']

        if self.config.RESUME.flag:
            self.resume()
        
        if self.config.FINETUNE.flag:
            self.fine_tune()
    

    def train(self,current_step):
        # Pytorch [N, C, D, H, W]
        # initialize
        start = time.time()
        self.A.train()
        self.B.train()
        self.C.train()
        self.Detector.eval()
        self.set_requires_grad(self.A, True)
        self.set_requires_grad(self.B, True)
        self.set_requires_grad(self.C, True)
        self.set_requires_grad(self.Detector, False)
        writer = self.kwargs['writer_dict']['writer']
        global_steps = self.kwargs['writer_dict']['global_steps_{}'.format(self.kwargs['model_type'])]

        # get the data
        data, _ = next(self._train_loader_iter)  # the core for dataloader
        self.data_time.update(time.time() - start)
        
        # base on the D to get each frame
        # in this method, D = 3 and not change
        future = data[:, :, -1, :, :].cuda() # t+1 frame 
        current = data[:, :, 1, :, :].cuda() # t frame
        past = data[:, :, 0, :, :].cuda() # t-1 frame

        bboxs = get_batch_dets(self.Detector, current)
        # this method is based on the objects to train the model insted of frames
        for index, bbox in enumerate(bboxs):
            if bbox.numel() == 0:
                bbox = bbox.new_zeros([1, 4])
            # get the crop objects
            input_currentObject_B, _ = multi_obj_grid_crop(current[index], bbox)
            future_object, _ = multi_obj_grid_crop(future[index], bbox)
            future2current = torch.stack([future_object, input_currentObject_B], dim=1)
            past_object, _ = multi_obj_grid_crop(past[index], bbox)
            current2past = torch.stack([input_currentObject_B, past_object], dim=1)

            _, _, input_objectGradient_A = frame_gradient(future2current)
            input_objectGradient_A = input_objectGradient_A.sum(1)
            _, _, input_objectGradient_C = frame_gradient(current2past)
            input_objectGradient_C = input_objectGradient_C.sum(1)
            # import ipdb; ipdb.set_trace()
            # True Process =================Start===================
            _, output_recGradient_A = self.A(input_objectGradient_A)
            _, output_recObject_B = self.B(input_currentObject_B)
            _, output_recGradient_C = self.C(input_objectGradient_C)
            # import ipdb; ipdb.set_trace()
            loss_A = self.a_loss(output_recGradient_A, input_objectGradient_A)
            loss_B = self.b_loss(output_recObject_B, input_currentObject_B)
            loss_C = self.c_loss(output_recGradient_C, input_objectGradient_C)

            loss_all = self.loss_lamada['A_loss'] * loss_A + self.loss_lamada['B_loss'] * loss_B + self.loss_lamada['C_loss'] * loss_C
            self.optim_ABC.zero_grad()
            loss_all.backward()
            self.optim_ABC.step()
            # record
            self.loss_meter_ABC.update(loss_all.detach())
            if self.config.TRAIN.general.scheduler.use:
                self.lr_abc.step()
        
            # ======================End==================

        self.batch_time.update(time.time() - start)

        if (current_step % self.steps.param['log'] == 0):
            msg = make_info_message(current_step, self.steps.param['max'], self.kwargs['model_type'], self.batch_time, self.config.TRAIN.batch_size, self.data_time, [self.loss_meter_ABC])
            self.logger.info(msg)
        writer.add_scalar('Train_loss_ABC', self.loss_meter_ABC.val, global_steps)

        if (current_step % self.steps.param['vis'] == 0):
            vis_objects = OrderedDict({
                'train_input_objectGradient_A': input_objectGradient_A.detach(),
                'train_input_currentObject_B': input_currentObject_B.detach(),
                'train_input_objectGradient_C': input_objectGradient_C.detach(),
                'train_output_recGradient_A': output_recGradient_A.detach(),
                'train_output_recObject_B': output_recObject_B.detach(),
                'train_output_recGradient_C': output_recGradient_C.detach()
            })
            # vis_objects['train_input_objectGradient_A'] =  input_objectGradient_A.detach()
            # vis_objects['train_input_currentObject_B'] =  input_currentObject_B.detach()
            # vis_objects['train_input_objectGradient_C'] = input_objectGradient_C.detach()
            # vis_objects['train_output_recGradient_A'] =  output_recGradient_A.detach()
            # vis_objects['train_output_recObject_B'] =  output_recObject_B.detach()
            # vis_objects['train_output_recGradient_C'] = output_recGradient_C.detach()
            tensorboard_vis_images(vis_objects, writer, global_steps, self.normalize.param['train'])
        global_steps += 1 
        # reset start
        start = time.time()
        
        self.saved_model = {'A':self.A, 'B':self.B, 'C':self.C}
        self.saved_optimizer = {'optim_ABC': self.optim_ABC}
        self.saved_loss = {'loss_ABC':self.loss_meter_ABC.val}
        self.kwargs['writer_dict']['global_steps_{}'.format(self.kwargs['model_type'])] = global_steps
    
    def mini_eval(self, current_step):
        if current_step % self.steps.param['mini_eval'] != 0:
            return
        temp_meter_A = AverageMeter()
        temp_meter_B = AverageMeter()
        temp_meter_C = AverageMeter()
        
        self.A.eval()
        self.B.eval()
        self.C.eval()
        self.Detector.eval()
        self.set_requires_grad(self.A, False)
        self.set_requires_grad(self.B, False)
        self.set_requires_grad(self.C, False)
        self.set_requires_grad(self.Detector, False)

        for data, _ in self.val_dataloader:
            # base on the D to get each frame
            # in this method, D = 3 and not change
            future_mini = data[:, :, -1, :, :].cuda() # t+1 frame 
            current_mini = data[:, :, 1, :, :].cuda() # t frame
            past_mini = data[:, :, 0, :, :].cuda() # t-1 frame

            bboxs_mini = get_batch_dets(self.Detector, current_mini)

            for index, bbox in enumerate(bboxs_mini):
                if bbox.numel() == 0:
                    bbox = bbox.new_zeros([1, 4])
                # get the crop objects
                input_currentObject_B, _ = multi_obj_grid_crop(current_mini[index], bbox)
                future_object, _ = multi_obj_grid_crop(future_mini[index], bbox)
                future2current = torch.stack([future_object, input_currentObject_B], dim=1)
                past_object, _ = multi_obj_grid_crop(past_mini[index], bbox)
                current2past = torch.stack([input_currentObject_B, past_object], dim=1)

                _, _, input_objectGradient_A = frame_gradient(future2current)
                input_objectGradient_A = input_objectGradient_A.sum(1)
                _, _, input_objectGradient_C = frame_gradient(current2past)
                input_objectGradient_C = input_objectGradient_C.sum(1)
            
                _, output_recGradient_A = self.A(input_objectGradient_A)
                _, output_recObject_B = self.B(input_currentObject_B)
                _, output_recGradient_C = self.C(input_objectGradient_C)

                psnr_A = psnr_error(output_recGradient_A.detach(), input_objectGradient_A)
                psnr_B = psnr_error(output_recObject_B.detach(), input_currentObject_B)
                psnr_C = psnr_error(output_recGradient_C.detach(), input_objectGradient_C)
                temp_meter_A.update(psnr_A.detach())
                temp_meter_B.update(psnr_B.detach())
                temp_meter_C.update(psnr_C.detach())

        self.logger.info(f'&^*_*^& ==> Step:{current_step}/{self.steps.param["max"]} the  A PSNR is {temp_meter_A.avg:.2f}, the B PSNR is {temp_meter_B.avg:.2f}, the C PSNR is {temp_meter_C.avg:.2f}')


class Inference(DefaultInference):
    NAME = ["OCAE.INFERENCE"]
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
        self.ovr_model_path = os.path.join(self.config.TRAIN.model_output, f'ocae_cfg@{self.config_name}#{self.verbose}.npy')

        model = defaults[0]
        if kwargs['parallel']:
            self.A = self.data_parallel(model['A'].load_state_dict(save_model['A']))
            self.B = self.data_parallel(model['B'].load_state_dict(save_model['B']))
            self.C = self.data_parallel(model['C'].load_state_dict(save_model['C']))
            self.Detector = self.data_parallel(model['Detector'])
        else:
            self.A = model['A'].load_state_dict(save_model['A']).cuda()
            self.B = model['B'].load_state_dict(save_model['B']).cuda()
            self.C = model['C'].load_state_dict(save_model['C']).cuda()
            self.Detector = model['Detector'].cuda()
        
        self.ovr_model = model['OVR']
        self.ovr_model = joblib.load(self.ovr_model_path)
        # self.load()

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
    