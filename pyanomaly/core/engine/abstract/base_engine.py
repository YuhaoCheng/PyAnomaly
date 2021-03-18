"""
@author:  Yuhao Cheng
@contact: yuhao.cheng[at]outlook.com
"""
import torch
import abc
import logging
from collections import OrderedDict, namedtuple

from pyanomaly.core.utils import AverageMeter, ParamSet
from ..utils import engine_save
from .abstract_engine import AbstractTrainer, AbstractInference, AbstractService

logger = logging.getLogger(__name__)

class BaseTrainer(AbstractTrainer):
    """The base class of trainers
    All of other methods' trainer must be the sub-class of this.
    """
    def __init__(self,**kwargs):
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
                model_dict: The directionary of the moddel
                dataloaders_dict: 
                optimizer_dict: 
                loss_function_dict:
                logger:
                cfg: 
                parallel=parallel_flag, 
                pretrain=cfg.MODEL.PRETRAINED.USE
                verbose=args.verbose, 
                time_stamp=time_stamp, 
                model_type=cfg.MODEL.NAME, 
                writer_dict=writer_dict, 
                config_name=cfg_name, 
                loss_lamada=loss_lamada,
                hooks=hooks, 
                evaluate_function=evaluate_function,
                lr_scheduler_dict=lr_scheduler_dict,
                final_output_dir=final_output_dir, 
                cpu=args.cpu
        """
        self._hooks = []
        # self._eval_hooks = []
        self._register_hooks(kwargs['hooks'])
        # logger & config
        # self.logger = defaults[4]
        self.config = kwargs['config']

        # devices
        self.engine_gpus = self.config.SYSTEM.gpus

         # set the configuration of the saving process
        save_cfg_template = namedtuple('save_cfg_template', ['output_dir', 'low',  'cfg_name', 'dataset_name', 'model_name', 'time_stamp'])
        self.save_cfg = save_cfg_template(output_dir=self.config.TRAIN.checkpoint_output, low=0.0, cfg_name=kwargs['config_name'], dataset_name=self.config.DATASET.name, model_name=self.config.MODEL.name, time_stamp=kwargs['time_stamp'])

        self.model = kwargs['model_dict']
        
        if kwargs['pretrain']:
            self.load_pretrain()
        
        dataloaders_dict = kwargs['dataloaders_dict']
        self._dataloaders_dict = dataloaders_dict
        self.train_dataloaders_dict = dataloaders_dict['train']
        self._train_loader_iter = iter(self.train_dataloaders_dict['general_dataset_dict']['all'])
        # temporal, but it is wrong !!!
        self.val_dataloaders_dict = dataloaders_dict['val']
        self.val_dataset_keys = list(dataloaders_dict['val']['general_dataset_dict'].keys())

        # get the optimizer
        self.optimizer = kwargs['optimizer_dict']

        # get the loss_fucntion
        self.loss_function = kwargs['loss_function_dict']

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
                                  train={'use':self.config.AUGMENT.train.normal.use, 'mean':self.config.AUGMENT.train.normal.mean, 'std':self.config.AUGMENT.train.normal.std}, 
                                  val={'use':self.config.AUGMENT.val.normal.use, 'mean':self.config.AUGMENT.val.normal.mean, 'std':self.config.AUGMENT.val.normal.std})

        self.steps = ParamSet(name='steps', log=self.config.TRAIN.log_step, vis=self.config.TRAIN.vis_step, eval=self.config.TRAIN.eval_step, save=self.config.TRAIN.save_step, 
                              max=self.config.TRAIN.max_steps, dynamic_steps=self.config.TRAIN.dynamic_steps)

        self.evaluate_function = kwargs['evaluate_function']
        
        # hypyer-parameters of loss
        self.loss_lamada = kwargs['loss_lamada']

        # the lr scheduler
        self.lr_scheduler_dict = kwargs['lr_scheduler_dict']

        # initialize the saved objects
        # self.saved_model = OrderedDict()
        # self.saved_optimizer = OrderedDict()
        # self.saved_loss = OrderedDict()
        self.saved_stuff = OrderedDict()

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
    
    def load_pretrain(self):
        """Load the pretrain model.
        Using this method to load the pretrain model or checkpoint. 
        This method only loads the model file, not loads optimizer and etc.

        Args:
            None
        Returns:
            None
        
        """
        model_path = self.config.MODEL.pretrain_model

        if  model_path is '':
            logger.info('=>Not have the pre-train model! Training from the scratch')
        else:
            logger.info(f'=>Loading the model in {model_path}')
            pretrain_model = torch.load(model_path)
            if 'epoch' in pretrain_model.keys():
                logger.info('(|_|) ==> Use the check point file')
                # self.model.load_state_dict(pretrain_model['model_state_dict'])
                # model_file = pretrain_model['model_state_dict']
                self._load_file(self.model.keys(), pretrain_model)
            else:
                logger.info('(+_+) ==> Use the model file')
                # self.model.load_state_dict(pretrain_model['state_dict'])
                # model_file = pretrain_model['state_dict']
                self._load_file(self.model.keys(), pretrain_model)
    
    def resume(self):
        """Load files used for resume training.
        The method loads the model file and the optimzier file.
        """
        logger.info('=> Resume the previous training')
        checkpoint_path = self.config.TRAIN.resume.checkpoint_path
        logger.info(f'=> Load the checkpoint from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path)
        # self.model.load_state_dict(checkpoint['model_state_dict'])
        self._load_file(self.model.keys(), checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._load_file(self.optimizer.keys(), checkpoint['optimizer_state_dict'])
    

    def fine_tune(self):
        """Set the fine-tuning layers
        This method will set the not fine-tuning layers freezon and the fine-tuning layers activate.

        """
        # need to improve
        layer_list = self.config.TRAIN.finetune.layer_list
        logger.info('=> Freeze layers except start with:{}'.format(layer_list))
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
        logger.info('<!_!> ==> Data Parallel')
        gpus = [int(i) for i in self.config.SYSTEM.gpus]
        model_parallel = torch.nn.DataParallel(model.cuda(), device_ids=gpus)
        return model_parallel
    
    
    def after_step(self, current_step):
        # acc = 0.0
        for h in self._hooks:
            h.after_step(current_step)

    def after_train(self):
        for h in self._hooks:
            h.after_train()
        
        self.save(self.config.TRAIN.max_steps, flag='final')

    def save(self, current_step, best=False, flag='inter'):
        """Save method.
        The method is used to save the model or checkpoint. The following attributes are related to this function.
            self.saved_stuff(dict): the dictionary of saving things, such as model, optimizer, loss, step. 
        Args:
            current_step(int): The current step. 
            best(bool): indicate whether is the best model

        """
        if best:
            # result_dict = engine_save(self.config, self.kwargs['config_name'], self.saved_stuff, current_step, self.kwargs['time_stamp'], self.accuarcy, flag='best', verbose=(self.kwargs['model_type'] + '#' + self.verbose), best=True, save_model=True)
            result_dict = engine_save(self.saved_stuff, current_step, self.accuarcy, save_cfg=self.save_cfg, flag=flag, verbose=(self.kwargs['model_type'] + '#' + self.verbose), best=True, save_model=True)
            self.result_path = result_dict['model_file']
        else:
            # result_dict = engine_save(self.config, self.kwargs['config_name'], self.saved_stuff, current_step, self.kwargs['time_stamp'], self.accuarcy, flag='best', verbose=(self.kwargs['model_type'] + '#' + self.verbose), best=False, save_model=False)
            result_dict = engine_save(self.saved_stuff, current_step, self.accuarcy, save_cfg=self.save_cfg, flag=flag, verbose=(self.kwargs['model_type'] + '#' + self.verbose), best=False, save_model=False)
    
        
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
    def __init__(self, **kwargs):
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
                model_dict: The directionary of the moddel
                dataloaders_dict: 
                optimizer_dict: 
                loss_function_dict:
                logger:
                cfg: 
                parallel=parallel_flag, 
                pretrain=cfg.MODEL.PRETRAINED.USE
                verbose=args.verbose, 
                time_stamp=time_stamp, 
                model_type=cfg.MODEL.NAME, 
                writer_dict=writer_dict, 
                config_name=cfg_name, 
                loss_lamada=loss_lamada,
                hooks=hooks, 
                evaluate_function=evaluate_function,
                lr_scheduler_dict=lr_scheduler_dict,
                final_output_dir=final_output_dir, 
                cpu=args.cpu
        """
        self._hooks = []
        # self._eval_hooks = []
        self._register_hooks(kwargs['hooks'])
        # logger & config
        # self.logger = defaults[4]
        self.config = kwargs['config']
        # devices
        self.engine_gpus = self.config.SYSTEM.gpus

        self.model = kwargs['model_dict']
        
        if kwargs['pretrain']:
            self.load_pretrain()
        
        dataloaders_dict = kwargs['dataloaders_dict']

        self.val_dataloaders_dict = dataloaders_dict['train']
        self.val_dataset_keys = list(dataloaders_dict['test']['general_dataset_dict'].keys())

        # get the optimizer
        self.optimizer = kwargs['optimizer_dict']

        # get the loss_fucntion
        self.loss_function = kwargs['loss_function_dict']

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

    def load_model(self, model_path):
        """Load the model from the model file.
        """
        logger.info(f'=>Loading the Test model in {model_path}')
        model_file = torch.load(model_path)
        self._load_file(self.model.keys(), model_file)


    @abc.abstractmethod
    def custom_setup(self):
        pass

    @abc.abstractmethod
    def inference(self, current_step):
        pass


class BaseService(AbstractService):
    """The BaseService class
    The 'service' means that the user only want to use the model to run on the real data instaed of the data from the dataset.
    So, in this class, it just provide the function to get the model, and regularize the pipeline to use the model. 
    """
    def __init__(self, **kwargs):
        """Initialization Method.
        Args:
            kwargs(dict): the default will have:
                model_dict: The directionary of the moddel
                config: 
                parallel=parallel_flag, 
                pretrain=cfg.MODEL.PRETRAINED.USE
                verbose=args.verbose, 
                time_stamp=time_stamp, 
                model_type=cfg.MODEL.NAME, 
                config_name=cfg_name, 
                hooks=hooks, 
                evaluate_function=evaluate_function,
                final_output_dir=final_output_dir, 
                cpu=args.cpu
        """
        self._hooks = []
        self._register_hooks(kwargs['hooks'])
        self.config = kwargs['config']

        # devices
        self.engine_gpus = self.config.SYSTEM.gpus

        self.model = kwargs['model_dict']
        
        if self.config.VAL.model_file == '':
            raise Exception("Not have the Trained model file")
        else:
            self.model_path = self.config.VAL.model_file

        # # get the optimizer
        # self.optimizer = kwargs['optimizer_dict']

        # # get the loss_fucntion
        # self.loss_function = kwargs['loss_function_dict']

        # basic meter, will be deprecated in the future.
        self.batch_time =  AverageMeter(name='batch_time')
        self.data_time = AverageMeter(name='data_time')


        # others
        self.verbose = kwargs['verbose']
        self.config_name = kwargs['config_name']
        self.kwargs = kwargs
        self.normalize = ParamSet(name='normalize', 
                                  train={'use':self.config.ARGUMENT.train.normal.use, 'mean':self.config.ARGUMENT.train.normal.mean, 'std':self.config.ARGUMENT.train.normal.std}, 
                                  val={'use':self.config.ARGUMENT.val.normal.use, 'mean':self.config.ARGUMENT.val.normal.mean, 'std':self.config.ARGUMENT.val.normal.std})

        self.evaluate_function = kwargs['evaluate_function']
        
        # hypyer-parameters of loss
        # self.loss_lamada = kwargs['loss_lamada']

        # the lr scheduler
        # self.lr_scheduler_dict = kwargs['lr_scheduler_dict']

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
        
        
        # get the losses
        for item_key in self.loss_function.keys():
            attr_name = str(item_key)
            self.__setattr__(attr_name, self.loss_function[attr_name])

        self.custom_setup()
        self.load_model(self.model_path)

    def load_model(self, model_path):
        """Load the model from the model file.
        """
        logger.info(f'=>Loading the Test model in {model_path}')
        model_file = torch.load(model_path)
        self._load_file(self.model.keys(), model_file)

    @abc.abstractmethod
    def custom_setup(self):
        pass

    @abc.abstractmethod
    def execute(self):
        pass
    