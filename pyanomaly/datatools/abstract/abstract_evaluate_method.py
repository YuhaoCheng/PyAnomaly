import abc
import six

__all__ = ['AbstractEvalMethod']

@six.add_metaclass(abc.ABCMeta)
class AbstractEvalMethod(object):

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.dataset_name = self.cfg.DATASET.name
        self.dataset_params = self.cfg.DATASET
        self.model_name = self.cfg.MODEL.name
    
    @abc.abstractmethod
    def load_ground_truth(self):
        '''
        Aim to load the gt of dataset.
        '''
        pass

    @abc.abstractmethod
    def load_results(self, result_file):
        pass
    
    @abc.abstractmethod
    def eval_method(self, result, gt ):
        '''
        The actual method to get the eval metrics
        '''
        pass

    @abc.abstractmethod
    def compute(self, result_file_dict):
        '''
        Aim to get the final result
        '''
        pass