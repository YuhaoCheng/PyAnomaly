import torch
import cv2
import glob
import os
import numpy as np
from collections import OrderedDict
from torch.utils.data import Dataset
# from pyanomaly.datatools.abstract.anomaly_video_dataset import AbstractVideoAnomalyDataset
# from pyanomaly.datatools.abstract.tools import ImageLoader, VideoLoader
from ..abstract.video_dataset import FrameLevelVideoDataset
from ..datatools_registry import DATASET_REGISTRY
import logging
logger = logging.getLogger(__name__)

# @DATASET_REGISTRY.register()
# class AvenuePedShanghaiAll(AbstractVideoAnomalyDataset):
#     _NAME = 'AvenuePedShanghai'
#     def _get_frames(self, video_name):
#         cusrsor = self.videos[video_name]['cursor']
#         if (cusrsor + self.clip_length) > self.videos[video_name]['length']:
#             cusrsor = 0
#         if self.mini:
#             rng = np.random.RandomState(2020)
#             start = rng.randint(0, self.videos[video_name]['length'] - self.clip_length)
#         else:
#             start = cusrsor

#         video_clip, video_clip_original = self.video_loader.read(self.videos[video_name]['frames'], start, start+self.clip_length, clip_length=self.sampled_clip_length, 
#                                                                  step=self.frame_step)
#         self.videos[video_name]['cursor'] = cusrsor + self.clip_step
#         return video_clip
    
    
#     def get_image(self, image_name):
#         # keep for debug
#         image =  self.image_loader.read(image_name)
#         return image

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
# #-----------------Functions Part-------------------------
# def _get_test_dataset(cfg, aug)->(list, list):
#     dataset_list = OrderedDict()    
#     video_dirs = os.listdir(cfg.DATASET.test_path)
#     video_dirs.sort()
#     for t_dir in video_dirs:
#         _temp_test_folder = os.path.join(cfg.DATASET.test_path, t_dir)
#         dataset = AvenuePedShanghaiAll(_temp_test_folder, clip_length=cfg.DATASET.test_clip_length, sampled_clip_length=cfg.DATASET.test_sampled_clip_length, 
#                                        clip_step=cfg.DATASET.test_clip_step, frame_step=cfg.DATASET.test_frame_step,transforms=aug, one_video=True, cfg=cfg, phase='val')
#         dataset_list[t_dir] = dataset
#     video_keys = list(dataset_list.keys())
#     return (dataset_list, video_keys)

# def _get_train_w_dataset(cfg, aug)->(list, list):
#     dataset_list = OrderedDict()
#     video_dirs = os.listdir(cfg.DATASET.train_path)
#     video_dirs.sort()
#     for t_dir in video_dirs:
#         _temp_test_folder = os.path.join(cfg.DATASET.train_path, t_dir)
#         if cfg.DATASET.name == 'shanghai':
#             dataset = AvenuePedShanghaiAll(_temp_test_folder, clip_length=cfg.DATASET.test_clip_length, sampled_clip_length=cfg.DATASET.test_sampled_clip_length, 
#                                            clip_step=cfg.DATASET.test_clip_step, frame_step=cfg.DATASET.test_frame_step, one_video=True, mini=True, transforms=aug, cfg=cfg, phase='train')
#             print(f'\033[1;31mUsing the MINI dataset of {cfg.DATASET.name} for calc the w \033[0m')
#         else:
#             dataset = AvenuePedShanghaiAll(_temp_test_folder, clip_length=cfg.DATASET.train_clip_length, sampled_clip_length=cfg.DATASET.train_sampled_clip_length, 
#                                            clip_step=cfg.DATASET.train_clip_step, frame_step=cfg.DATASET.train_frame_step,transforms=aug, one_video=True, cfg=cfg, phase='train')
#         dataset_list[t_dir] = dataset
#     video_keys = list(dataset_list.keys())
#     return (dataset_list, video_keys)

# def _get_cluster_dataset(cfg, aug)->(list, list):
#     dataset_list = OrderedDict()
    
#     video_dirs = os.listdir(cfg.DATASET.train_path)
#     video_dirs.sort()
#     for t_dir in video_dirs:
#         _temp_train_folder = os.path.join(cfg.DATASET.train_path, t_dir)
#         dataset = AvenuePedShanghaiAll(_temp_train_folder, clip_length=cfg.DATASET.train_clip_length, sampled_clip_length=cfg.DATASET.train_sampled_clip_length, 
#                                        clip_step=cfg.DATASET.train_clip_step, frame_step=cfg.DATASET.train_frame_step,is_training=True, transforms=aug, one_video=True, cfg=cfg, phase='train')
#         dataset_list[t_dir] = dataset
#     video_keys = list(dataset_list.keys())
#     return (dataset_list, video_keys)

# def get_avenue_ped_shanghai(cfg, flag, aug):
#     '''
#     Using the function to register the Dataset
#     '''
#     if flag == 'train':
#         # import ipdb; ipdb.set_trace()
#         t = AvenuePedShanghaiAll(cfg.DATASET.train_path, clip_length=cfg.DATASET.train_clip_length, sampled_clip_length=cfg.DATASET.train_sampled_clip_length, 
#                                  frame_step=cfg.DATASET.train_frame_step, clip_step=cfg.DATASET.train_clip_step, transforms=aug, cfg=cfg, phase='train')
#     elif flag == 'val':
#         t = AvenuePedShanghaiAll(cfg.DATASET.test_path, clip_length=cfg.DATASET.test_clip_length, sampled_clip_length=cfg.DATASET.test_sampled_clip_length, 
#                                  clip_step=cfg.DATASET.test_clip_step, frame_step=cfg.DATASET.test_frame_step, transforms=aug, mini=True, cfg=cfg, phase='train')
#         print(f'\033[1;31mUsing the MINI dataset of {cfg.DATASET.name} for evaluation in the process of training\033[0m')
#     elif flag == 'test':
#         t = _get_test_dataset(cfg, aug)
#     elif flag == 'train_w':
#         t = _get_train_w_dataset(cfg, aug)
#     elif flag == 'cluster_train':
#         t = _get_cluster_dataset(cfg, aug)
#     return t 


if __name__ == '__main__':
    import ipdb; ipdb.set_trace()