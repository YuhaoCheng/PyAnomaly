"""
@author:  Yuhao Cheng
@contact: yuhao.cheng[at]outlook.com
"""
import torch
from pyanomaly.core.utils import AverageMeter, ParamSet
from ..utils import engine_save_checkpoint
from ..utils import engine_save_model
from .abstract_engine import AbstractTrainer, AbstractInference
import abc
from collections import OrderedDict
# import logging
# logger = logging.getLogger(__name__)

class BaseTrainer(AbstractTrainer):
    """The base class of trainers
    All of other methods' trainer must be the sub-class of this.
    """
    def __init__(self, *defaults, **kwargs):
        """Initialization Method.
        Args:
            defaults(tuple): the default will have:
                0 0->model:{'Generator':net_g, 'Driscriminator':net_d, 'FlowNet':net_flow}
                - 1->train_dataloader: the dataloader    # Will be deprecated in the future
                - 2->val_dataloader: the dataloader     # Will be deprecated in the future
                1 -->dataloader_dict: the dict of all the dataloader will be used in the process
                2 3->optimizer:{'optimizer_g':op_g, 'optimizer_d'}
                3 4->loss_function: {'g_adverserial_loss':.., 'd_adverserial_loss':..., 'gradient_loss':.., 'opticalflow_loss':.., 'intentsity_loss':.. }
                4 5->logger: the logger of the whole training process
                5 6->config: the config object of the whole process

            kwargs(dict): the default will have:
                verbose(str):
                parallel(bool): True-> data parallel
                pertrain(bool): True-> use the pretarin model
                dataloaders_dict: will to replace the train_dataloader and test_dataloader
                hooks
        """
        self._hooks = []
        # self._eval_hooks = []
        self._register_hooks(kwargs['hooks'])
        # logger & config
        self.logger = defaults[4]
        self.config = defaults[5]

        self.model = defaults[0]
        
        if kwargs['pretrain']:
            self.load_pretrain()
        # =============================the old version to get iter of dataloader========================
        # self.train_dataloader = defaults[1]
        # self._train_loader_iter = iter(self.train_dataloader)

        # self.val_dataloader = defaults[2]
        # self._val_loader_iter = iter(self.val_dataloader)
        # ==============================================================================================
        dataloaders_dict = defaults[1]
        self._dataloaders_dict = dataloaders_dict
        self.train_dataloaders_dict = dataloaders_dict['train']
        self._train_loader_iter = iter(self.train_dataloaders_dict['general_dataset_dict']['all'])
        # self.val_dataloaders_dict = dataloaders_dict['test']
        # temporal, but it is wrong !!!
        # self._val_loader_iter = iter(self.train_dataloaders_dict['general_dataset_dict']['video_datasets']['all'])
        # self._val_loader_iter = iter(self.train_dataloaders_dict['general_dataset_dict']['all'])
        self.val_dataloaders_dict = dataloaders_dict['val']
        self.val_dataset_keys = list(dataloaders_dict['val']['general_dataset_dict'].keys())
        # get the optimizer
        self.optimizer = defaults[2]

        # get the loss_fucntion
        self.loss_function = defaults[3]

        # basic meter
        self.batch_time =  AverageMeter(name='batch_time')
        self.data_time = AverageMeter(name='data_time')

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
                              max=self.config.TRAIN.max_steps, dynamic_steps=self.config.TRAIN.dynamic_steps)

        self.evaluate_function = kwargs['evaluate_function']
        
        # hypyer-parameters of loss
        self.loss_lamada = kwargs['loss_lamada']

        # the lr scheduler
        self.lr_scheduler_dict = kwargs['lr_scheduler_dict']

        # initialize the saved objects
        self.saved_model = OrderedDict()
        self.saved_optimizer = OrderedDict()
        self.saved_loss = OrderedDict()

        # Get the models
        for item_key in self.model.keys():
            attr_name = str(item_key)
            if self.kwargs['parallel']:
                temp_model = self.data_parallel(self.model[item_key])
            else:
                temp_model = self.model[item_key].cuda()
            self.__setattr__(attr_name, temp_model)
        
        # get the optimizer
        for item_key in self.optimizer.keys():
            attr_name = str(item_key)
            # get the optimizer
            self.__setattr__(attr_name, self.optimizer[item_key])
            # get the lr scheduler
            self.__setattr__(f'{attr_name}_scheduler', self.lr_scheduler_dict[f'{attr_name}_scheduler'])
        
        # get the losses
        for item_key in self.loss_function.keys():
            attr_name = str(item_key)
            self.__setattr__(attr_name, self.loss_function[attr_name])

        self.custom_setup()

        # Continue training a model from a checkpoint
        if self.config.TRAIN.resume.use:
            self.resume()
        
        # Fine-tine a trained model
        if self.config.TRAIN.finetune.use:
            self.fine_tune()
    
    def _load_file(self, model_keys, model_file):
        """Method to load the data into pytorch structure.
        Args:
            model_keys: The keys of model
            model_file: The data of the model
        """
        for item in model_keys:
            item = str(item)
            getattr(self, item).load_state_dict(model_file[item]['state_dict'])
        self.logger.info('Finish load!')

    def load_pretrain(self):
        model_path = self.config.MODEL.pretrain_model

        if  model_path is '':
            self.logger.info('=>Not have the pre-train model! Training from the scratch')
        else:
            self.logger.info(f'=>Loading the model in {model_path}')
            pretrain_model = torch.load(model_path)
            if 'epoch' in pretrain_model.keys():
                self.logger.info('(|_|) ==> Use the check point file')
                # self.model.load_state_dict(pretrain_model['model_state_dict'])
                # model_file = pretrain_model['model_state_dict']
                self._load_file(self.model.keys(), pretrain_model)
            else:
                self.logger.info('(+_+) ==> Use the model file')
                # self.model.load_state_dict(pretrain_model['state_dict'])
                # model_file = pretrain_model['state_dict']
                self._load_file(self.model.keys(), pretrain_model)
    
    def resume(self):
        self.logger.info('=> Resume the previous training')
        checkpoint_path = self.config.TRAIN.resume.checkpoint_path
        self.logger.info(f'=> Load the checkpoint from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path)
        # self.model.load_state_dict(checkpoint['model_state_dict'])
        self._load_file(self.model.keys(), checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._load_file(self.optimizer.keys(), checkpoint['optimizer_state_dict'])
    
    def fine_tune(self):
        # need to improve
        layer_list = self.config.TRAIN.finetune.layer_list
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
        """Parallel the models.
        Data parallel the model by using torch.nn.DataParallel
        Args:
            model: torch.nn.Module
        Returns:
            model_parallel
        """
        self.logger.info('<!_!> ==> Data Parallel')
        gpus = [int(i) for i in self.config.SYSTEM.gpus]
        model_parallel = torch.nn.DataParallel(model.cuda(), device_ids=gpus)
        return model_parallel
    
    def set_all(self, is_train):
        """Set all train or eval.
        Set all of models in this trainer in eval or train model
        Args:
            is_train: bool. True=train mode; False=eval mode
        Returns:
            None
        """
        for item in self.model.keys():
            self.set_requires_grad(getattr(self, str(item)), is_train)
            if is_train:
                getattr(self, str(item)).train()
            else:
                getattr(self, str(item)).eval()
    
    def after_step(self, current_step):
        # acc = 0.0
        for h in self._hooks:
            h.after_step(current_step)

    def after_train(self):
        for h in self._hooks:
            h.after_train()
        
        self.save(self.config.TRAIN.max_steps)

    def save(self, current_step, best=False):
        """Save method.
        The method is used to save the model or checkpoint. The following attributes are related to this function.
            self.saved_model: the model or a dict of combination of models
            self.saved_optimizer:  the optimizer or a dict of combination of optimizers
            self.saved_loss: the loss  or a dict of the combination  of loss

        Args:
            current_step(int): The current step. 
            best(bool): indicate whether is the best model

        """
        if best:
            engine_save_checkpoint(self.config, self.kwargs['config_name'], self.saved_model, current_step, self.saved_loss, self.saved_optimizer, self.logger, self.kwargs['time_stamp'], self.accuarcy, flag='best', verbose=(self.kwargs['model_type'] + '#' + self.verbose),best=best)
            self.result_path = engine_save_model(self.config, self.kwargs['config_name'], self.saved_model, self.logger, self.kwargs['time_stamp'], self.accuarcy, verbose=(self.kwargs['model_type'] + '#' + self.verbose), best=best)
        else:
            engine_save_checkpoint(self.config, self.kwargs['config_name'], self.saved_model, current_step, self.saved_loss, self.saved_optimizer, self.logger, self.kwargs['time_stamp'], self.accuarcy, verbose=(self.kwargs['model_type'] + '#' + self.verbose), best=best)

    @abc.abstractmethod
    def custom_setup(self):
        """Extra setup method.
        This method help users to define some extra methods
        """
        pass

    @abc.abstractmethod
    def train(self,current_step):
        """Actual training function.
        Re-write by sub-class to implement the training functions.

        Args:
            current_step(int): The current step
        """
        pass
    
 
class BaseInference(AbstractInference):
    def __init__(self, *defaults, **kwargs):
        """
        Args:
            defaults(tuple): the default will have:
                0 0->model:{'Generator':net_g, 'Driscriminator':net_d, 'FlowNet':net_flow}
                - 1->train_dataloader: the dataloader    # Will be deprecated in the future
                - 2->val_dataloader: the dataloader     # Will be deprecated in the future
                1 -->dataloader_dict: the dict of all the dataloader will be used in the process
                2 3->optimizer:{'optimizer_g':op_g, 'optimizer_d'}
                3 4->loss_function: {'g_adverserial_loss':.., 'd_adverserial_loss':..., 'gradient_loss':.., 'opticalflow_loss':.., 'intentsity_loss':.. }
                4 5->logger: the logger of the whole training process
                5 6->config: the config object of the whole process

            kwargs(dict): the default will have:
                verbose(str):
                parallel(bool): True-> data parallel
                pertrain(bool): True-> use the pretarin model
                dataloaders_dict: will to replace the train_dataloader and test_dataloader
                extra param:
                    test_dataset_keys: the dataset keys of each video
                    test_dataset_dict: the dataset dict of whole test videos
        """
        self._hooks = []
        # self._eval_hooks = []
        self._register_hooks(kwargs['hooks'])
        # logger & config
        self.logger = defaults[4]
        self.config = defaults[5]

        self.model = defaults[0]
        
        if kwargs['pretrain']:
            self.load_pretrain()
        
        dataloaders_dict = defaults[1]

        self.val_dataloaders_dict = dataloaders_dict['train']
        self.val_dataset_keys = list(dataloaders_dict['test']['general_dataset_dict'].keys())

        # get the optimizer
        self.optimizer = defaults[2]

        # get the loss_fucntion
        self.loss_function = defaults[3]

        # basic meter
        self.batch_time =  AverageMeter(name='batch_time')
        self.data_time = AverageMeter(name='data_time')

        # others
        self.verbose = kwargs['verbose']
        self.accuarcy = 0.0  # to store the accuracy varies from epoch to epoch
        self.config_name = kwargs['config_name']
        self.result_path = ''
        self.kwargs = kwargs
        self.normalize = ParamSet(name='normalize', 
                                  train={'use':self.config.ARGUMENT.train.normal.use, 'mean':self.config.ARGUMENT.train.normal.mean, 'std':self.config.ARGUMENT.train.normal.std}, 
                                  val={'use':self.config.ARGUMENT.val.normal.use, 'mean':self.config.ARGUMENT.val.normal.mean, 'std':self.config.ARGUMENT.val.normal.std})

        self.steps = ParamSet(name='steps', log=self.config.VAL.log_step, vis=self.config.VAL.vis_step, eval=self.config.TRAIN.eval_step, max=self.config.TRAIN.max_steps)

        self.evaluate_function = kwargs['evaluate_function']
        
        # hypyer-parameters of loss
        self.loss_lamada = kwargs['loss_lamada']

        # the lr scheduler
        self.lr_scheduler_dict = kwargs['lr_scheduler_dict']

        # initialize the saved objects
        # None

        # Get the models
        for item_key in self.model.keys():
            attr_name = str(item_key)
            if self.kwargs['parallel']:
                temp_model = self.data_parallel(self.model[item_key])
            else:
                temp_model = self.model[item_key].cuda()
            self.__setattr__(attr_name, temp_model)
        
        # get the optimizer
        # None
        
        # get the losses
        for item_key in self.loss_function.keys():
            attr_name = str(item_key)
            self.__setattr__(attr_name, self.loss_function[attr_name])

        self.custom_setup()

    
    
    def _load_file(self, model_keys, model_file):
        """Method to load the data into pytorch structure.
        Args:
            model_keys: The keys of model
            model_file: The data of the model
        """
        for item in model_keys:
            item = str(item)
            getattr(self, item).load_state_dict(model_file[item]['state_dict'])
        self.logger.info('Finish load!')

    def load_pretrain(self):
        model_path = self.config.MODEL.pretrain_model

        if  model_path is '':
            self.logger.info('=>Not have the pre-train model! Training from the scratch')
        else:
            self.logger.info(f'=>Loading the model in {model_path}')
            pretrain_model = torch.load(model_path)
            if 'epoch' in pretrain_model.keys():
                self.logger.info('(|_|) ==> Use the check point file')
                # self.model.load_state_dict(pretrain_model['model_state_dict'])
                # model_file = pretrain_model['model_state_dict']
                self._load_file(self.model.keys(), pretrain_model)
            else:
                self.logger.info('(+_+) ==> Use the model file')
                # self.model.load_state_dict(pretrain_model['state_dict'])
                # model_file = pretrain_model['state_dict']
                self._load_file(self.model.keys(), pretrain_model)


    # def load(self):
    #     if type(self.model) == type(dict()):
    #         for k, v in self.model.items():
    #             temp = torch.load(self.model_path)
    #             if k[0] == 'F':
    #                 continue
    #             self.model[k].load_state_dict(temp[k[0]])
    #     else:
    #         self.model.load_state_dict(torch.load(self.model_path))
    def set_all(self, is_train):
        """
        Set all of models in this trainer in eval or train model
        Args:
            is_train: bool. True=train mode; False=eval mode
        Returns:
            None
        """
        for item in self.trainer.model.keys():
            self.set_requires_grad(getattr(self, str(item)), is_train)
            if is_train:
                getattr(self, str(item)).train()
            else:
                getattr(self, str(item)).eval()
    

    def data_parallel(self, model):
        '''
        Data parallel the model
        '''
        self.logger.info('<!_!> ==> Data Parallel')
        gpus = [int(i) for i in self.config.SYSTEM.gpus]
        model_parallel = torch.nn.DataParallel(model, device_ids=gpus).cuda()
        return model_parallel


    @abc.abstractmethod
    def custom_setup(self):
        pass

    @abc.abstractmethod
    def inference(self, current_step):
        pass
    