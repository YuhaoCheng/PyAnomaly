"""
@author:  Yuhao Cheng
@contact: yuhao.cheng[at]outlook.com
"""
from ..abstract.video_dataset import FrameLevelVideoDataset
from ..datatools_registry import DATASET_REGISTRY
import logging
logger = logging.getLogger(__name__)

@DATASET_REGISTRY.register()
class Avenue(FrameLevelVideoDataset):
    def custom_setup(self):
        # logger.info('the name of the avenue')
        self.dataset_class_name = 'Avenue'

@DATASET_REGISTRY.register()
class Ped2(FrameLevelVideoDataset):
    def custom_setup(self):
        # logger.info('the name of the ped2')
        self.dataset_class_name = 'Ped2'

@DATASET_REGISTRY.register()
class Shanghai(FrameLevelVideoDataset):
    def custom_setup(self):
        # logger.info('the name of the Shanghai')
        self.dataset_class_name = 'Shanghai'
