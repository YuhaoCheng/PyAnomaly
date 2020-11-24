from ..datasets_registry import DATASET_FACTORY_REGISTRY
from ..datasets_registry import DATASET_REGISTRY
from .avenue_ped_shanghai import *
from ..abstract.abstract_dataset_factory import AbstractDatasetFactory, GetWDataset
from collections import OrderedDict, namedtuple
import os
# @DATASET_FACTORY_REGISTRY.registry()
# class AvenueFactory(object):
#     NORMAL = ['stae', 'amc']
#     CLASS1 = ['ocae']
#     CLASS2 = ['memae']
#     def __init__(self, cfg, aug) -> None:
#         self.cfg = cfg
#         self.model_name = cfg.MODEL.name
#         self.ingredient = DATASET_REGISTRY.get('AvenuePedShanghaiAll')
#         self.aug = aug

#     def _build_normal(self):
#         """
#         """
#         return 0
#     def _build_class1(self):
#         """
#         """
#         return 0

#     def _build_class2(self):
#         """
#         """
#         return 0

#     def _build(self):
#         if self.model_name in AvenueFactory.NORMAL:
#             dataloader = self._build_normal()
#         elif self.model_name in AvenueFactory.CLASS1:
#             dataloader = self._build_class1()
#         elif self.model_name in AvenueFactory.CLASS2:
#             dataloader = self._build_class2()
#         else:
#             raise Exception('123')
        
#         return dataloader
    
#     def __call__(self):
#         dataset_dict = self._build()
#         return dataset_dict

@DATASET_FACTORY_REGISTRY.registry()
class VideoAnomalyDatasetFactory(AbstractDatasetFactory, GetWDataset):
    NORMAL = ['stae', 'amc', 'anopcn', 'anopred']
    CLASS1 = ['ocae']
    CLASS2 = ['memae']
    def __init__(self, cfg, aug, is_training=True) -> None:
        # self.cfg = cfg
        # self.model_name = cfg.MODEL.name
        # self.aug = aug
        super(VideoAnomalyDatasetFactory, self).__init__(cfg, aug, is_training)
        self.ingredient = DATASET_REGISTRY.get(self.dataset_name)

    def _produce_train_dataset(self):
        train_dataset_dict = OrderedDict()
        train_dataset = self.ingredient(self.dataset_params.train_path, clip_length=self.dataset_params.train_clip_length, 
                                        sampled_clip_length=self.dataset_params.train_sampled_clip_length, 
                                        frame_step=self.dataset_params.train_frame_step,clip_step=self.dataset_params.train_clip_step, is_training=True,
                                        transforms=self.aug, cfg=self.cfg)
        train_dataset_dict['video_keys'] = 'all'
        train_dataset_dict['video_datasets'] = train_dataset
        # pass 
        return train_dataset_dict
    
    def _produce_test_dataset(self):
        dataset_dict = OrderedDict()
        video_dirs = os.listdir(self.dataset_params.test_path)
        video_dirs.sort()
        for video_dir in video_dirs:
            _temp_test_folder = os.path.join(self.dataset_params.test_path, video_dir)
            dataset = self.ingredient(_temp_test_folder, clip_length=self.dataset_params.test_clip_length, 
                                      sampled_clip_length=self.dataset_params.test_sampled_clip_length, 
                                      clip_step=self.dataset_params.test_clip_step, frame_step=self.dataset_params.test_frame_step, is_training=False,
                                      transforms=self.aug, one_video=True, cfg=self.cfg)
            dataset_dict[video_dir] = dataset
        video_keys = list(dataset_dict.keys())
        test_dataset_dict = OrderedDict()
        test_dataset_dict['video_keys'] = video_keys
        test_dataset_dict['video_datasets'] = dataset_dict
        return test_dataset_dict
        # pass
    
    def _build_normal(self):
        """
        """
        return 0
    
    def _build_class1(self):
        """
        """
        return 0

    def _build_class2(self):
        """
        """
        return 0
    
    def _build(self):
        if self.model_name in VideoAnomalyDatasetFactory.NORMAL:
            dataloader = self._build_normal()
        elif self.model_name in VideoAnomalyDatasetFactory.CLASS1:
            dataloader = self._build_class1()
        elif self.model_name in VideoAnomalyDatasetFactory.CLASS2:
            dataloader = self._build_class2()
        else:
            raise Exception('123')
        
        return dataloader
    
    def __call__(self):
        dataset_dict = self._build()
        return dataset_dict

def _get_test_dataset(cfg, aug)->(list, list):
    dataset_list = OrderedDict()    
    video_dirs = os.listdir(cfg.DATASET.test_path)
    video_dirs.sort()
    for t_dir in video_dirs:
        _temp_test_folder = os.path.join(cfg.DATASET.test_path, t_dir)
        dataset = AvenuePedShanghaiAll(_temp_test_folder, clip_length=cfg.DATASET.test_clip_length, sampled_clip_length=cfg.DATASET.test_sampled_clip_length, 
                                    clip_step=cfg.DATASET.test_clip_step, frame_step=cfg.DATASET.test_frame_step,transforms=aug, one_video=True, cfg=cfg, phase='val')
        dataset_list[t_dir] = dataset
    video_keys = list(dataset_list.keys())
    return (dataset_list, video_keys)