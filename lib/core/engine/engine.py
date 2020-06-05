import os
import sys
sys.path.append('../../')
import torch
import time
from PIL import Image
from torch.autograd import Variable
import torchvision.transforms.functional as tf
from lib.utils.utils import save_checkpoint
from lib.utils.utils import save_model
from lib.core.utils import AverageMeter

from .abstract_engine import AbstractTrainer, AbstractInference

class BaseTrainer(AbstractTrainer):
    '''
    Define the basic things about training
    '''
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, loss_function, logger,config, verbose='None',parallel=True, pretrain=False, **kwargs):
        # logger & config
        self.logger = logger
        self.config = config
        
        # basic things
        if parallel:
            self.model = self.data_parallel(model)
        else:
            self.model = model
        
        if pretrain:
            self.load_pretrain()

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.loss_function = loss_function

        # basic meter
        self.batch_time =  AverageMeter()
        self.data_time = AverageMeter()
        self.loss_basic = AverageMeter()

        # others
        self.verbose = verbose
        self.accuarcy = 0.0  # to store the accuracy varies from epoch to epoch
        self.kwargs = kwargs
        self.total_steps = len(self.train_dataloader)
        self.result_path = ''

        if self.config.RESUME.flag:
            self.resume()
        
        if self.config.FINETUNE.flag:
            self.fine_tune()

    def _get_time(self):
        '''
        Get the current time
        '''
        return time.strftime('%Y-%m-%d-%H-%M') # 2019-08-07-10-34
    
    def load_pretrain(self):
        model_path = self.config.MODEL.pretrain_model
        if  model_path is '':
            self.logger.info('=>Not have the pre-train model! Training from the scratch')
        else:
            self.logger.info('=>Loading the model in {}'.format(model_path))
            pretrain_model = torch.load(model_path)
            if 'epoch' in pretrain_model.keys():
                self. logger.info('(|_|) ==> Use the check point file')
                self.model.load_state_dict(pretrain_model['model_state_dict'])
            else:
                self.logger.info('(+_+) ==> Use the model file')
                # model_all.load_state_dict(pretrain_model['state_dict'], strict=False)
                self.model.load_state_dict(pretrain_model['state_dict'])
    
    def resume(self):
        self.logger.info('=> Resume the previous training')
        checkpoint_path = self.config.RESUME.checkpoint_path
        self.logger.info('=> Load the checkpoint from {}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def fine_tune(self):
        layer_list = self.config.FINETUNE.layer_list
        self.logger.info('=> Freeze layers except start with:{}'.format(layer_list))
        for n, p in self.model.named_parameters():
            parts = n.split('.')
            # consider the data parallel situation
            if parts[0] == 'module':
                if parts[1] not in layer_list:
                    p.requires_grad = False
                if p.requires_grad:
                    print(n)
            else:
                if parts[0] not in layer_list:
                    p.requires_grad = False
                if p.requires_grad:
                    print(n)
        self.logger.info('Finish Setting freeze layers')
    
    def data_parallel(self, model):
        '''
        Data parallel the model
        '''
        self.logger.info('<!_!> ==> Data Parallel')
        gpus = [int(i) for i in self.config.SYSTEM.gpus]
        model_parallel = torch.nn.DataParallel(model, device_ids=gpus).cuda()
        return model_parallel

    def run(self, current_epoch):
        '''
        Run the whole process:
        1. print the log information
        2. execute training process
        3. evaluate(including the validation and test)
        4. save model
        '''
        self.logger.info('-0_0- ==>|{}| Start Traing the {}/{} epoch'.format(self._get_time(), current_epoch, self.config.TRAIN.epochs))
        
        # train the model
        self.train(current_epoch)

        # evaluate 
        acc = self.evaluate(current_epoch)
        if acc > self.accuarcy:
            self.accuarcy = acc
            # save the model & checkpoint
            self.save(current_epoch, best=True)
        else:
            # save the checkpoint
            self.save(current_epoch)
            self.logger.info('LOL==>the accuracy is not imporved in epcoh{}'.format(current_epoch))


    def save(self, current_epoch, best=False):
        if best:
            save_checkpoint(self.config, self.kwargs['config_name'], self.model, current_epoch, self.loss_basic.val, self.optimizer, self.logger, self.kwargs['time_stamp'], self.accuarcy, flag='best', verbose=(self.kwargs['cae_type'] + '#' + self.verbose))
            self.result_path = save_model(self.config, self.kwargs['config_name'], self.model, self.logger, self.kwargs['time_stamp'], self.accuarcy, verbose=(self.kwargs['cae_type'] + '#' + self.verbose))
        else:
            save_checkpoint(self.config, self.kwargs['config_name'], self.model, current_epoch, self.loss_basic.val, self.optimizer, self.logger, self.kwargs['time_stamp'], self.accuarcy, verbose=(self.kwargs['cae_type'] + '#' + self.verbose))
        

    def train(self,current_epoch):
        start = time.time()
        self.model.train()
        writer = self.kwargs['writer_dict']['writer']
        global_steps = self.kwargs['writer_dict']['global_steps_{}'.format(self.kwargs['cae_type'])]

        for step, (data,target) in enumerate(self.train_dataloader):
            self.data_time.update(time.time() - start)
            # True Process
            output = self.model(data)
            loss = self.loss_function(output, target.cuda())
            self.optimizer.zero_grad()
            loss.sum().backward()
            self.optimizer.step()
            # record
            self.loss_basic.update(loss.sum())
            self.batch_time.update(time.time() - start)

            if (step % 10 == 0) or (step == self.total_steps - 1):
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                    'Type: {cae_type}\t' \
                    'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                    'Speed: {speed:.1f} samples/s\t' \
                    'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                    'Loss: {losses.val:.5f} ({losses.avg:.5f})\t'.format(current_epoch, step, self.total_steps, cae_type=self.kwargs['cae_type'], batch_time=self.batch_time, speed=self.config.TRAIN.batch_size/self.batch_time.val, data_time=self.data_time,losses=self.loss_basic)
                self.logger.info(msg)
            writer.add_scalar('Train_loss', self.loss_basic.val, global_steps)
            global_steps += 1 
            # reset start
            start = time.time()

        self.kwargs['writer_dict']['global_steps_{}'.format(self.kwargs['cae_type'])] = global_steps
    
    def evaluate(self, current_epoch):
        '''
        Evaluate the results of the model
        !!! Will change, e.g. accuracy, mAP.....
        !!! Or can call other methods written by the official
        '''
        self.model.eval()
        correct_1 = 0.0
        correct_5 = 0.0
        for _ ,(data, target) in enumerate(self.val_dataloader):
            data = Variable(data).cuda()
            target = Variable(target).cuda()
            score = self.model(data)
            _, pred = score.topk(5,1,largest=True, sorted=True)

            target = target.view(target.size(0), -1).expand_as(pred)
            correct = pred.eq(target).float()

            # compute top5
            correct_5 += (correct[:, :5].sum()) / len(self.val_dataloader.dataset)

            # compute top1
            correct_1 += (correct[:, :1].sum()) / len(self.val_dataloader.dataset)
        
        self.logger.info('&^*_*^& ==> Epoch:{}/{} the top1 is {}, the top5 is {}'.format(current_epoch, self.config.TRAIN.epochs,correct_1, correct_5))

        return correct_1

        
class BaseInference(AbstractInference):
    def __init__(self, model, model_path,logger, config, verbose='None',parallel=True, mode='dataset',**kwargs):
        '''
        Args:
            mode: change the mode of inference, can choose: dataset | image
        '''
        self.logger = logger
        self.config = config
        self.model_path = model_path
        if parallel:
            self.model = self.data_parallel(model)
        else:
            self.model = model
        self.load()

        self.verbose = verbose
        self.kwargs = kwargs
        self.mode = mode

        self.metric = 0.0

        if self.mode == 'dataset':
            self.inference_dataloader = self.kwargs['inference_dataloader']

    def load(self):
        self.model.load_state_dict(torch.load(self.model_path))

    def data_parallel(self, model):
        '''
        Data parallel the model
        '''
        self.logger.info('<!_!> ==> Data Parallel')
        gpus = [int(i) for i in self.config.SYSTEM.gpus]
        model_parallel = torch.nn.DataParallel(model, device_ids=gpus).cuda()
        return model_parallel
    
    def run(self, **kwargs):
        if self.mode == 'dataset':
            metric = self.evaluate()
        elif self.mode == 'image':
            self.get_result(kwargs)
        else:
            raise Exception('Wrong inference mode')
    
    def evaluate(self):
        '''
        Evaluate the results of the model
        !!! Will change, e.g. accuracy, mAP.....
        !!! Or can call other methods written by the official
        '''
        self.model.eval()
        correct_1 = 0.0
        correct_5 = 0.0
        for _ ,(data, target) in enumerate(self.inference_dataloader):
            data = Variable(data).cuda()
            target = Variable(target).cuda()
            score = self.model(data)
            _, pred = score.topk(5,1,largest=True, sorted=True)

            target = target.view(target.size(0), -1).expand_as(pred)
            correct = pred.eq(target).float()

            # compute top5
            correct_5 += (correct[:, :5].sum()) / len(self.inference_dataloader.dataset)

            # compute top1
            correct_1 += (correct[:, :1].sum()) / len(self.inference_dataloader.dataset)
        
        self.logger.info('Inference ==> the top1 is {}, the top5 is {}'.format(correct_1, correct_5))

        return correct_1
    
    def get_result(self, kwargs):
        '''
        Get the results for one image
        '''
        self.model.eval()
        image = Image.open(kwargs['image'])
        image = tf.resize(kwargs['size'])
        image = tf.to_tensor(image)
        
        



