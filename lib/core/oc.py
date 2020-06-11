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

from lib.core.engine.default_engine import DefaultTrainer, DefaultInference
from lib.core.utils import AverageMeter, multi_obj_grid_crop, frame_gradient, flow_batch_estimate
from lib.datatools.evaluate.utils import psnr_error


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
            # self.G = model['Generator'].to(torch.device('cuda:0'))
            # self.D = model['Discriminator'].to(torch.device('cuda:1'))
            self.C = self.data_parallel(model['C'])
            self.Detector = self.data_parallel(model['Detector'])
        else:
            self.A = model['A'].cuda()
            self.B = model['B'].cuda()
            self.C = model['C'].cuda()
            self.Detector = model['Detector'].cuda()
        
        if kwargs['pretrain']:
            self.load_pretrain()

        self.train_dataloader = defaults[1]
        self._train_loader_iter = iter(self.train_dataloader)

        self.val_dataloader = defaults[2]
        self._val_loader_iter = iter(self.val_dataloader)

        # get the optimizer
        optimizer = defaults[3]
        # self.optim_A = optimizer['optimizer_a']
        # self.optim_B = optimizer['optimizer_b']
        # self.optim_C = optimizer['optimizer_c']
        self.optim_ABC = optimizer['optimizer_abc']

        # get the loss_fucntion
        loss_function = defaults[4]
        self.a_loss = loss_function['A_loss']
        self.b_loss = loss_function['B_loss']
        self.c_loss = loss_function['C_loss']

        # basic meter
        self.batch_time =  AverageMeter()
        self.data_time = AverageMeter()
        self.loss_meter_A = AverageMeter()
        self.loss_meter_B = AverageMeter()
        self.loss_meter_C = AverageMeter()
        self.loss_meter_ABC = AverageMeter()
        self.psnr = AverageMeter()

        # others
        self.verbose = kwargs['verbose']
        self.accuarcy = 0.0  # to store the accuracy varies from epoch to epoch
        self.config_name = kwargs['config_name']
        self.kwargs = kwargs
        # self.total_steps = len(self.train_dataloader)
        self.result_path = ''
        self.log_step = self.config.TRAIN.log_step # how many the steps, we will show the information
        self.eval_step = self.config.TRAIN.eval_step 
        self.save_step = self.config.TRAIN.save_step # save the model whatever the acc of the model
        self.max_steps = self.config.TRAIN.max_steps
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
        # self.lr_b = kwargs['lr_shechulder_b']
        # self.lr_c = kwargs['lr_shechulder_c']

        if self.config.RESUME.flag:
            self.resume()
        
        if self.config.FINETUNE.flag:
            self.fine_tune()
    
    def get_batch_dets(self, batch_image):
        """
        Use the detecron2
        """
        image_list = list()
        batch_size = batch_image.size(0)
        images = torch.chunk(batch_image, batch_size, dim=0)
        for image in images:
            image_list.append({"image":image.squeeze_(0).mul(255).byte()[[2,0,1],:,:]})
        outputs = self.Detector(image_list)
        # import ipdb; ipdb.set_trace()
        bboxs = []
        frame_objects = OrderedDict()
        max_objects = 0
        min_objects = 1000
        for frame_id, out in enumerate(outputs):
            temp = out['instances'].pred_boxes.tensor.detach()
            temp.requires_grad = False
            frame_objects[frame_id] = temp.size(0)
            if frame_objects[frame_id] > max_objects:
                max_objects = frame_objects[frame_id]
            if frame_objects[frame_id] < min_objects:
                min_objects = frame_objects[frame_id]
            bboxs.append(temp)
        
        # bboxs = torch.stack(bboxs, dim=0)
        # self.logger.info(f'the max object:{max_objects}, the min objects:{min_objects}')

        return bboxs


    def train(self,current_step):
        start = time.time()
        self.A.train()
        self.B.train()
        self.C.train()
        self.Detector.eval()
        writer = self.kwargs['writer_dict']['writer']
        global_steps = self.kwargs['writer_dict']['global_steps_{}'.format(self.kwargs['model_type'])]

        data = next(self._train_loader_iter)  # the core for dataloader
        self.data_time.update(time.time() - start)

        future = data[:, -1, :, :, :].cuda() # t+1 frame 
        current = data[:, 1, :, :, :].cuda() # t frame
        past = data[:, 0, :, :, :].cuda() # t frame

        # bbox = self.Detector(current)
        bboxs = self.get_batch_dets(current)
        for index, bbox in enumerate(bboxs):
            # import ipdb; ipdb.set_trace()
            if bbox.numel() == 0:
                bbox = bbox.new_zeros([1, 4])
                # print('NO objects')
                # continue
            # import ipdb; ipdb.set_trace()
            current_object, _ = multi_obj_grid_crop(current[index], bbox)
            future_object, _ = multi_obj_grid_crop(future[index], bbox)
            future2current = torch.stack([future_object, current_object], dim=1)
            past_object, _ = multi_obj_grid_crop(past[index], bbox)
            current2past = torch.stack([current_object, past_object], dim=1)

            _, _, A_input = frame_gradient(future2current)
            A_input = A_input.sum(1)
            _, _, C_input = frame_gradient(current2past)
            C_input = C_input.sum(1)
            # import ipdb; ipdb.set_trace()
            # True Process =================Start===================
            #---------update optim_G ---------
            _, A_output = self.A(A_input)
            # pred_flow_esti_tensor = torch.cat([input_last, G_output_flow],1)
            # gt_flow_esti_tensor = torch.cat([input_last, target], 1)
            # flow_gt,_ = self.batch_estimate(gt_flow_esti_tensor)
            # flow_pred = self.batch_estimate(pred_flow_esti_tensor)
            _, B_output = self.B(current_object)
            _, C_output = self.C(C_input)
            # import ipdb; ipdb.set_trace()
            loss_A = self.a_loss(A_output, A_input)
            loss_B = self.b_loss(B_output, current_object)
            loss_C = self.c_loss(C_output, C_input)

            loss_all = self.loss_lamada['A_loss'] * loss_A + self.loss_lamada['B_loss'] * loss_B + self.loss_lamada['C_loss'] * loss_C
            self.optim_ABC.zero_grad()
            loss_all.backward()
            self.optim_ABC.step()
            # record
            self.loss_meter_ABC.update(loss_all.detach())
            if self.config.TRAIN.general.scheduler.use:
                self.lr_abc.step()
            # #---------update optim_D ---------------
            # self.optim_D.zero_grad()
            # # G_output_flow,  G_output_frame = self.G(input)
            # real_sigmoid, real = self.D(torch.cat([target,G_output_flow.detach()],dim=1))
            # fake_sigmoid, fake = self.D(torch.cat([G_output_frame.detach(),G_output_flow.detach()], dim=1))
            # # loss_d = self.d_adv_loss(self.D(torch.cat([input,target],dim=1)), self.D(torch.cat(G_output_frame.detach()))
            # loss_d_1 = self.d_adv_loss_1(real_sigmoid, torch.ones_like(real))
            # loss_d_2 = self.d_adv_loss_2(fake_sigmoid, torch.zeros_like(fake))
            # loss_d = self.loss_lamada['amc_d_adverserial_loss_1'] * loss_d_1 + self.loss_lamada['amc_d_adverserial_loss_2'] * loss_d_2
            # # loss_d.sum().backward()
            # # import ipdb; ipdb.set_trace()
            # loss_d.backward()

            # self.optim_D.step()
            # if self.config.TRAIN.scheduler.use:
            #     self.lr_d.step()
            # self.loss_meter_D.update(loss_d.detach())
            # ======================End==================

        self.batch_time.update(time.time() - start)

        if (current_step % self.log_step == 0):
            msg = 'Step: [{0}/{1}]\t' \
                'Type: {cae_type}\t' \
                'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                'Speed: {speed:.1f} samples/s\t' \
                'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                'Loss_ABC: {losses_ABC.val:.5f} ({losses_ABC.avg:.5f})\t'.format(current_step, self.max_steps, cae_type=self.kwargs['model_type'], batch_time=self.batch_time, speed=self.config.TRAIN.batch_size/self.batch_time.val, data_time=self.data_time,losses_ABC=self.loss_meter_ABC)
            self.logger.info(msg)
        writer.add_scalar('Train_loss_ABC', self.loss_meter_ABC.val, global_steps)
        global_steps += 1 
        # reset start
        start = time.time()
        # del data, input, input_last, loss_d, loss_g_all, target
        # torch.cuda.empty_cache()
        
        self.saved_model = {'A':self.A, 'B':self.B, 'C':self.C}
        self.saved_optimizer = {'optim_ABC': self.optim_ABC}
        self.saved_loss = {'loss_ABC':self.loss_meter_ABC.val}
        self.kwargs['writer_dict']['global_steps_{}'.format(self.kwargs['model_type'])] = global_steps
    
    def mini_eval(self, current_step):
        pass

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
        
        model = defaults[0]
        if kwargs['parallel']:
            self.A = self.data_parallel(model['A'])
            self.B = self.data_parallel(model['B'])
            # self.G = model['Generator'].to(torch.device('cuda:0'))
            # self.D = model['Discriminator'].to(torch.device('cuda:1'))
            self.C = self.data_parallel(model['C'])
            self.Detector = self.data_parallel(model['Detector'])
        else:
            self.A = model['A'].cuda()
            self.B = model['B'].cuda()
            self.C = model['C'].cuda()
            self.Detector = model['Detector'].cuda()
        
        # self.load()

        self.verbose = kwargs['verbose']
        self.kwargs = kwargs
        self.config_name = kwargs['config_name']
        # self.mode = kwargs['mode']

        self.test_dataset_keys = kwargs['test_dataset_keys']
        self.test_dataset_dict = kwargs['test_dataset_dict']

        self.test_dataset_keys_w = kwargs['test_dataset_keys_w']
        self.test_dataset_dict_w = kwargs['test_dataset_dict_w']
        self.metric = 0.0

       
    
    def inference(self):
        for h in self._hooks:
            h.inference()
    