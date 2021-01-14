"""
@author:  Yuhao Cheng
@contact: yuhao.cheng[at]outlook.com
"""
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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from ..abstract.base_engine import BaseTrainer, BaseInference
from pyanomaly.core.utils import AverageMeter, multi_obj_grid_crop, frame_gradient, get_batch_dets, tensorboard_vis_images, ParamSet, make_info_message
from pyanomaly.datatools.evaluate.utils import psnr_error
from ..engine_registry import ENGINE_REGISTRY

try:
    from sklearn.externals import joblib
except:
    import joblib

__all__ = ['OCAETrainer', 'OCAEInference']

@ENGINE_REGISTRY.register()
class OCAETrainer(BaseTrainer):
    NAME = ["OCAE.TRAIN"]
    def custom_setup(self):
        self.ovr_model = OneVsRestClassifier(LinearSVC(random_state = 0), n_jobs=16)

        # basic meter
        self.loss_meter_ABC = AverageMeter(name='loss_ABC')

        self.ovr_model_path = os.path.join(self.config.TRAIN.model_output, f'ocae_cfg@{self.config_name}#{self.verbose}.npy') 
        # import ipdb; ipdb.set_trace()

    def train(self,current_step):
        # Pytorch [N, C, D, H, W]
        # initialize
        start = time.time()
        self.set_requires_grad(self.A, True)
        self.set_requires_grad(self.B, True)
        self.set_requires_grad(self.C, True)
        self.set_requires_grad(self.Detector, False)
        self.A.train()
        self.B.train()
        self.C.train()
        self.Detector.eval()
        writer = self.kwargs['writer_dict']['writer']
        global_steps = self.kwargs['writer_dict']['global_steps_{}'.format(self.kwargs['model_type'])]

        # get the data
        data, anno, meta = next(self._train_loader_iter)  # the core for dataloader
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
            # future2current = torch.stack([future_object, input_currentObject_B], dim=1)
            current2future = torch.stack([input_currentObject_B, future_object], dim=1)
            past_object, _ = multi_obj_grid_crop(past[index], bbox)
            # current2past = torch.stack([input_currentObject_B, past_object], dim=1)
            past2current = torch.stack([past_object, input_currentObject_B], dim=1)

            _, _, input_objectGradient_A = frame_gradient(current2future)
            input_objectGradient_A = input_objectGradient_A.sum(1)
            _, _, input_objectGradient_C = frame_gradient(past2current)
            input_objectGradient_C = input_objectGradient_C.sum(1)
            # import ipdb; ipdb.set_trace()
            # True Process =================Start===================
            # original_A = (0.3 * input_objectGradient_A[:,0] + 0.59 * input_objectGradient_A[:,1] + 0.11 * input_objectGradient_A[:,2]).unsqueeze(1)
            # original_B = (0.3 * input_currentObject_B[:,0] + 0.59 * input_currentObject_B[:,1] + 0.11 * input_currentObject_B[:,2]).unsqueeze(1)
            # original_C = (0.3 * input_objectGradient_C[:,0] + 0.59 * input_objectGradient_C[:,1] + 0.11 * input_objectGradient_C[:,2]).unsqueeze(1)
            _, output_recGradient_A, original_A = self.A(input_objectGradient_A)
            _, output_recObject_B, original_B = self.B(input_currentObject_B)
            _, output_recGradient_C, original_C = self.C(input_objectGradient_C)
            # import ipdb; ipdb.set_trace()
            # loss_A = self.a_loss(output_recGradient_A, input_objectGradient_A)
            # loss_B = self.b_loss(output_recObject_B, input_currentObject_B)
            # loss_C = self.c_loss(output_recGradient_C, input_objectGradient_C)
            loss_A = self.ALoss(output_recGradient_A, original_A)
            loss_B = self.BLoss(output_recObject_B, original_B)
            loss_C = self.CLoss(output_recGradient_C, original_C)

            loss_all = self.loss_lamada['ALoss'] * loss_A + self.loss_lamada['BLoss'] * loss_B + self.loss_lamada['CLoss'] * loss_C
            self.optimizer_ABC.zero_grad()
            loss_all.backward()
            self.optimizer_ABC.step()
            # record
            self.loss_meter_ABC.update(loss_all.detach())
            if self.config.TRAIN.general.scheduler.use:
                self.optimizer_ABC_scheduler.step()
            
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
            tensorboard_vis_images(vis_objects, writer, global_steps, self.normalize.param['train'])
        global_steps += 1 
        # reset start
        start = time.time()
        
        # self.saved_model = {'A':self.A, 'B':self.B, 'C':self.C}
        # self.saved_model['A'] = self.A
        # self.saved_model['B'] = self.B
        # self.saved_model['C'] = self.C
        # # self.saved_optimizer = {'optim_ABC': self.optim_ABC}
        # self.saved_optimizer['optimizer_ABC'] = self.optimizer_ABC
        # # self.saved_loss = {'loss_ABC':self.loss_meter_ABC.val}
        # self.saved_loss['loss_ABC'] = self.loss_meter_ABC.val

        self.saved_stuff['step'] = global_steps
        self.saved_stuff['loss'] = self.loss_meter_ABC.val
        self.saved_stuff['A'] = self.A
        self.saved_stuff['B'] = self.B
        self.saved_stuff['C'] = self.C
        self.saved_stuff['optimizer_ABC'] = self.optimizer_ABC

        self.kwargs['writer_dict']['global_steps_{}'.format(self.kwargs['model_type'])] = global_steps
    

@ENGINE_REGISTRY.register()
class OCAEInference(BaseInference):
    NAME = ["OCAE.INFERENCE"]
    def custom_setup(self):
        self.ovr_model_path = os.path.join(self.config.TRAIN.model_output, f'ocae_cfg@{self.config_name}#{self.verbose}.npy')
        self.ovr_model = self.model['OVR']
        self.ovr_model = joblib.load(self.ovr_model_path)

    def inference(self):
        for h in self._hooks:
            h.inference()
    