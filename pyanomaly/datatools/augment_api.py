import torch
import torchvision.transforms as T
from .augment.augment_builder import AugmentBuilder

class AugmentAPI(AugmentBuilder):
    def __init__(self,cfg, logger):
        super(AugmentAPI, self).__init__(cfg, logger)
    
    def add(self, extra_aug):
        '''
        add the extra aug into the aug
        '''
        print('add the extra_aug')

    def __call__(self, flag='train'):
        super(AugmentAPI, self)._get_node(flag)
        # import ipdb; ipdb.set_trace()
        t = super(AugmentAPI, self)._build()
        return t