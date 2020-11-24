from torch.utils.data import DataLoader
class AbstractBuilder(object):
    # def __init__(self, cfg):
    #     '''
    #     the init of the builder
    #     '''
    #     self.cfg = cfg
    
    def build(self)->DataLoader:
        '''
        the method to build the dataloader
        the building process includes three parts:
            1. dataset
            2. sampler
            3. collect_fn
        ===> dataloader
        '''
        raise Exception('No implement')
    
    def _build_dataset(self):
        '''
        build the dataset
        '''
        raise Exception('No implement')
    
    def _build_sampler(self):
        '''
        build the sampler
        '''
        pass

    def _build_collect_fn(self):
        '''
        build the collect fn
        '''
        pass