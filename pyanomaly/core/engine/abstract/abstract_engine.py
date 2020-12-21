"""
@author:  Yuhao Cheng
@contact: yuhao.cheng[at]outlook.com
"""
import time
import weakref
from pyanomaly.core.hook.abstract import HookBase
import abc

class AbstractEngine(object):
    """Abstract engine class.
    The defination of an engine. Containing some basic and fundanmental methods
    """
    def __init__(self):
        """Initialization Method.
        """
        self._hooks = []
    
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


