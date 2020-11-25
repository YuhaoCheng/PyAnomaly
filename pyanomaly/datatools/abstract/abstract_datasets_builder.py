from torch.utils.data import DataLoader
import six
import abc

@six.add_metaclass(abc.ABCMeta)
class AbstractBuilder(object):
    # def __init__(self, cfg):
    #     '''
    #     the init of the builder
    #     '''
    #     self.cfg = cfg

    @abc.abstractmethod
    def build(self)->DataLoader:
        '''
        the method to build the dataloader
        the building process includes three parts:
            1. dataset
            2. sampler
            3. collect_fn
        ===> dataloader
        '''
        # raise Exception('No implement')
        pass

    @abc.abstractmethod
    def _build_dataset(self):
        '''
        build the dataset
        '''
        # raise Exception('No implement')
        pass

    @abc.abstractmethod
    def _build_sampler(self):
        '''
        build the sampler
        '''
        pass

    # @abc.abstractmethod
    def _build_collect_fn(self):
        '''
        build the collect fn
        '''
        pass