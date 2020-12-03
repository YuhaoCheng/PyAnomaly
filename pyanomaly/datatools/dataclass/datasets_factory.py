"""
@author:  Yuhao Cheng
@contact: yuhao.cheng[at]outlook.com
"""
from ..datatools_registry import DATASET_FACTORY_REGISTRY
from ..datatools_registry import DATASET_REGISTRY
from .avenue_ped_shanghai import *
from ..abstract import AbstractDatasetFactory, GetWDataset, GetClusterDataset
from collections import OrderedDict, namedtuple
import os
__all__ = ['VideoAnomalyDatasetFactory']

@DATASET_FACTORY_REGISTRY.register()
class VideoAnomalyDatasetFactory(AbstractDatasetFactory, GetWDataset, GetClusterDataset):
    NORMAL = ['stae', 'amc', 'anopcn', 'anopred']
    NEED_W = ['memae']
    NEED_CLUSTER = ['ocae']
    def __init__(self, cfg, aug_dict, is_training=True) -> None:
        super(VideoAnomalyDatasetFactory, self).__init__(cfg, aug_dict, is_training)
        self.aug_dict = aug_dict
        self.ingredient = DATASET_REGISTRY.get(self.dataset_name)
        self._jude_need_cluster()
        self._jude_need_w()
    
    def _jude_need_w(self):
        if self.model_name in VideoAnomalyDatasetFactory.NEED_W:
            self.need_w_flag = True
        else:
            self.need_w_flag = False

    def _jude_need_cluster(self):
        if self.model_name in VideoAnomalyDatasetFactory.NEED_CLUSTER:
            self.need_cluster_flag = True
        else:
            self.need_cluster_flag = False

    def _produce_train_dataset(self):
        train_dataset_dict = OrderedDict()
        train_dataset = self.ingredient(self.dataset_params.train.data_path, clip_length=self.dataset_params.train.clip_length, 
                                        sampled_clip_length=self.dataset_params.train.sampled_clip_length, 
                                        frame_step=self.dataset_params.train.frame_step,clip_step=self.dataset_params.train.clip_step, is_training=True,
                                        transforms=self.aug_dict['train_aug'], cfg=self.cfg)
        train_dataset_dict['video_keys'] = ['all']
        train_dataset_dict['video_datasets'] = OrderedDict()
        train_dataset_dict['video_datasets']['all'] = train_dataset
        return train_dataset_dict
    
    def _produce_val_dataset(self):
        dataset_dict = OrderedDict()
        video_dirs = os.listdir(self.dataset_params.val.data_path)
        video_dirs.sort()
        for video_dir in video_dirs:
            _temp_test_folder = os.path.join(self.dataset_params.val.data_path, video_dir)
            dataset = self.ingredient(_temp_test_folder, clip_length=self.dataset_params.val.clip_length, 
                                      sampled_clip_length=self.dataset_params.val.sampled_clip_length, 
                                      clip_step=self.dataset_params.val.clip_step, frame_step=self.dataset_params.val.frame_step, is_training=False,
                                      transforms=self.aug_dict['test_aug'], one_video=True, cfg=self.cfg)
            dataset_dict[video_dir] = dataset
        video_keys = list(dataset_dict.keys())
        test_dataset_dict = OrderedDict()
        test_dataset_dict['video_keys'] = video_keys
        test_dataset_dict['video_datasets'] = dataset_dict
        return test_dataset_dict
    
    def _produce_w_dataset(self):
        dataset_dict = OrderedDict()
        video_dirs = os.listdir(self.dataset_params.train.data_path)
        video_dirs.sort()
        for video_dir in video_dirs:
            _temp_folder = os.path.join(self.dataset_params.train.data_path, video_dir)
            dataset = self.ingredient(_temp_folder, clip_length=self.dataset_params.train.clip_length, sampled_clip_length=self.dataset_params.train.clip_length, 
                                        clip_step=self.dataset_params.train.clip_length, frame_step=self.dataset_params.train.clip_length, one_video=True, is_training=True, 
                                        transforms=self.aug_dict['train_aug'], cfg=self.cfg)
            dataset_dict[video_dir] = dataset
        video_keys = list(dataset_dict.keys())
        w_dataset_dict = OrderedDict()
        w_dataset_dict['video_keys'] = video_keys
        w_dataset_dict['video_datasets'] = dataset_dict
        return w_dataset_dict

    def _produce_cluster_dataset(self):
        dataset_dict = OrderedDict()
        video_dirs = os.listdir(self.dataset_params.train_path)
        video_dirs.sort()
        for video_dir in video_dirs:
            _temp_folder = os.path.join(self.dataset_params.train.data_path, video_dir)
            dataset = self.ingredient(_temp_folder, clip_length=self.dataset_params.train.clip_length, sampled_clip_length=self.dataset_params.train.sampled_clip_length, 
                                       clip_step=self.dataset_params.train.clip_step, frame_step=self.dataset_params.train.frame_step, is_training=True, 
                                       transforms=self.aug_dict['train_aug'], one_video=True, cfg=self.cfg)
            dataset_dict[video_dir] = dataset
        video_keys = list(dataset_dict.keys())
        cluster_dataset_dict = OrderedDict()
        cluster_dataset_dict['video_keys'] = video_keys
        cluster_dataset_dict['video_datasets'] = dataset_dict
        return cluster_dataset_dict
    

    def _build(self):
        dataset_dict = OrderedDict()
        test_dataset_dict = self._produce_test_dataset()
        dataset_dict['test_dataset_dict'] = OrderedDict()
        dataset_dict['test_dataset_dict']['general_dataset_dict'] = test_dataset_dict

        if self.is_training:
            train_dataset_dict = self._produce_train_dataset()
            dataset_dict['train_dataset_dict'] = OrderedDict()
            dataset_dict['train_dataset_dict']['general_dataset_dict']  = train_dataset_dict
            if self.need_w_flag:
                w_dataset = self._produce_w_dataset()
                dataset_dict['train_dataset_dict']['w_dataset_dict'] = w_dataset
            if self.need_cluster_flag:
                cluster_dataset = self._produce_cluster_dataset()
                dataset_dict['train_dataset_dict']['cluster_dataset_dict'] = cluster_dataset

        return dataset_dict

    def __call__(self):
        dataset_dict = self._build()
        return dataset_dict
