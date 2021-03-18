"""
@author:  Yuhao Cheng
@contact: yuhao.cheng[at]outlook.com
"""
import time
import torch
import weakref
from collections import OrderedDict
from pyanomaly.core.hook.abstract import HookBase
import abc

import logging
logger = logging.getLogger(__name__)

class AbstractEngine(object):
    """Abstract engine class.
    The defination of an engine. Containing some basic and fundanmental methods
    """
    def __init__(self):
        """Initialization Method.
        """
        self._hooks = [] # the hooks of the engine
        self.engine_gpus = [] # the list of gpus which will be used in the parallel

    def _get_time(self):
        """Get the current time.
        
        """
        return time.strftime('%Y-%m-%d-%H-%M') # 2019-08-07-10-34
    
    def _register_hooks(self, hooks):
        """Register hooks to the trainer.
        The hooks are executed in the order they are registered.
        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.engine = weakref.proxy(self)
        self._hooks.extend(hooks)
    
    def set_requires_grad(self, nets, requires_grad=False):
        """Change the state of gradient.
        Args:
            nets(list): a list of networks
            requores_grad(bool): whether the networks require gradients or not  
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    
    def set_all(self, is_train):
        """Set all train or eval.
        Set all of models in this trainer in eval or train model
        self.model is the model dict from the outside.
        Args:
            is_train(bool): True=train mode; False=eval mode
        Returns:
            None
        """
        for item in self.model.keys():
            # import ipdb; ipdb.set_trace()
            self.set_requires_grad(getattr(self, str(item)), is_train)
            if is_train:
                getattr(self, str(item)).train()
            else:
                getattr(self, str(item)).eval()
    
    def data_parallel(self, model):
        """Parallel the models.
        Data parallel the model by using torch.nn.DataParallel
        Args:
            model: torch.nn.Module
        Returns:
            model_parallel
        """
        logger.info('<!_!> ==> Data Parallel')
        gpus = [int(i) for i in self.engine_gpus]
        model_parallel = torch.nn.DataParallel(model.cuda(), device_ids=gpus)
        return model_parallel


    def _load_file(self, model_keys, model_file):
        """Method to load the data into pytorch structure.
        This function satifies many situations
        Args:
            model_keys: The keys of the trainer's model
            model_file: The data of the model
        """
        def load_state(item, saved_model_file, parallel=False):
            # import ipdb; ipdb.set_trace()
            temp_list = list(saved_model_file.keys())
            temp = temp_list[0]
            logger.info(f'=> The first key in model file is {temp}')
            # import ipdb; ipdb.set_trace()
            if not parallel:
                # if the trainer NOT uses the data parallel to train
                if temp.startswith('module.'):
                    # if the saved model file is in the Dataparallel 
                    new_state_dict = OrderedDict()
                    for k, v in saved_model_file.items():
                        name = k[7:]
                        new_state_dict[name] = v
                    logger.info('(|+_+|) => Change the DataParallel Model File to Normal')
                    getattr(self, item).load_state_dict(new_state_dict)
                else:
                    getattr(self, item).load_state_dict(saved_model_file)
            else:
                # if the trainer uses the data parallel to train
                if not temp.startswith('module.'):
                    # if the saved model file is in the Dataparallel
                    new_state_dict = OrderedDict()
                    for k, v in saved_model_file.items():
                        # name = k[7:]
                        name = 'module.' + k
                        new_state_dict[name] = v
                    logger.info('(|+_+|) => Change the Normal Model File to DataParallel')
                    getattr(self, item).load_state_dict(new_state_dict)
                else:
                    getattr(self, item).load_state_dict(saved_model_file)
        
        # import ipdb; ipdb.set_trace()
        parallel_flag = self.kwargs['parallel']
        for item in model_keys:
            item = str(item)
            if 'state_dict' in model_file.keys():
                logger.info('\033[5;31m!! Directly use the file, the state_dict is in file\033[0m')
                # getattr(self, item).load_state_dict(model_file['state_dict'])
                load_state(item, model_file['state_dict'], parallel_flag)
                continue

            if item not in model_file.keys():
                # The name of the layer is the key in model file. For example:
                # {'conv1.weight':....., 'conv1.bias':.....}
                logger.info('\033[5;31m!! Directly use the file\033[0m')
                # getattr(self, item).load_state_dict(model_file)
                load_state(item, model_file, parallel_flag)
            elif item in model_file.keys():
                if 'state_dict' in model_file[item].keys():
                    # The name of the 
                    # getattr(self, item).load_state_dict(model_file[item]['state_dict'])
                    load_state(item, model_file[item]['state_dict'], parallel_flag)
                else:
                    logger.info('\033[5;31m!! Not have the state_dict \033[0m')
                    # getattr(self, item).load_state_dict(model_file[item])
                    load_state(item, model_file[item], parallel_flag)
           
        logger.info('Finish load the weights into the networks!')

class AbstractTrainer(AbstractEngine):
    """Abstract trainer class.
    The abstract defination of method and frame work during the training process. All of trainers must be the sub-class of this class.
    """
    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        """Initialization Method.
        """
        # self._hooks = []
        pass
    
    def run(self, start_iter, max_iter):
        """Start the training process.

        Define the order to execute the function in a tainer.
        The total number of the training steps is (max_iter - start_iter)

        Args:
            start_iter(int): The strat step number
            max_iter(int): The end step number.
        """
        self.before_train()
        for i in range(start_iter, max_iter):
            self.before_step(i)
            self.train(i)
            self.after_step(i)
        self.after_train()

    def train(self, current_step):
        """The single step of training.
        Args:
            current_step(int): Indicate the present step
        """
        pass
    
    def before_step(self, current_step):
        """The fucntion is execute before the train step.
        Args:
            current_step(int): Indicate the present step
        """
        pass

    def after_step(self, current_step):
        """The fucntion is execute after the train step.
        Args:
            current_step(int): Indicate the present step
        """
        pass

    def before_train(self):
        """The fucntion is excuted before the whole train function.
        Args:
            current_step(int): Indicate the present step
        """
        pass
    
    def after_train(self):
        """The fucntion is excuted after the whole train function.
        Args:
            current_step(int): Indicate the present step
        """
        pass

class AbstractInference(AbstractEngine):
    """Abstract inference class.
    The abstract defination of method and frame work during the inference process. All of inference must be the sub-class of this class.
    """
    @abc.abstractmethod
    def __init__(self, *args,**kwargs):
        """Initialization Method.
        """
        pass
    

    def run(self):
        """Start the inference process.

        Define the order to execute the function in a inference.
        """
        self.before_inference()
        self.inference()
        self.after_inference()
    
    def before_inference(self):
        """The fucntion is excuted before the whole inference function.
        """
        pass

    def inference(self):
        """Executation of the inference function.
        """
        pass

    def after_inference(self):
        """The fucntion is excuted after the whole inference function.
        """
        pass


class AbstractService(AbstractEngine):
    """Abstract Service class.
    The abstract defination of method and frame work during the service process. All of inference must be the sub-class of this class.
    """
    @abc.abstractmethod
    def __init__(self, *args,**kwargs):
        """Initialization Method.
        """
        pass
    
    def execute(self):
        """Start the inference process.

        Define the order to execute the function in a inference.
        """
        pass