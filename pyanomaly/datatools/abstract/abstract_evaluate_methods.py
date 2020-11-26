import abc
import six

@six.add_metaclass(abc.ABCMeta)
class AbstractEvalMethods(object):
    
    @abc.abstractmethod
    def load_ground_truth(self):
        pass
    
    @abc.abstractmethod
    def compute(self):
        pass