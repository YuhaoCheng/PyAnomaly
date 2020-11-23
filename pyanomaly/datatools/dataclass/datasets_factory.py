from ..datasets_registry import DATASET_FACTORY_REGISTRY
from ..datasets_registry import DATASET_REGISTRY
from .avenue_ped_shanghai import *


@DATASET_FACTORY_REGISTRY.registry()
class AvenueFactory(object):
    NORMAL = ['stae', 'amc']
    CLASS1 = ['ocae']
    CLASS2 = ['memae']
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.model_name = cfg.MODEL.name
        self.ingredient = DATASET_REGISTRY.get('AvenuePedShanghaiAll')

    def _build_normal(self):
        """
        """
        return 0
    def _build_class1(self):
        return 0
    def _build_class2(self):
        return 0
    
    def _build(self):
        
        if self.model_name in AvenueFactory.NORMAL:
            dataloader = self._build_normal()
        elif self.model_name in AvenueFactory.CLASS1:
            dataloader = self._build_class1()
        elif self.model_name in AvenueFactory.CLASS2:
            dataloader = self._build_class2()
        else:
            raise Exception('123')
        
        return dataloader
    
    def __call__(self):
        dataset_dict = self._build()
        return dataset_dict
        