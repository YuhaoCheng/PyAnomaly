import torch
import glob
import os
import numpy as np
from collections import OrderedDict
from .video_dataset import AbstractVideoDataset
from pyanomaly.datatools.abstract.tools import ImageLoader, VideoLoader

class AbstractVideoAnomalyDataset(AbstractVideoDataset):
    _NAME = 'AbstractVideoAnomalyDataset'
    def __init__(self, frames_folder, clip_length, sampled_clip_length, frame_step=1, clip_step=1, video_format='.mp4', fps=10, transforms=None, is_training=True, one_video=False, only_frame=True, mini=False, extra=False, cfg=None, phase='train', **kwargs):
        '''
        size = (h, w)
        is_training: True-> only get the frames, False-> get the frame and annotations
        '''
        super(AbstractVideoAnomalyDataset, self).__init__(frames_folder, clip_length, sampled_clip_length, frame_step=frame_step, clip_step=clip_step, video_format=video_format, fps=fps, 
                                                          transforms=transforms, is_training=is_training, one_video=one_video, mini=mini, cfg=cfg, phase='train', **kwargs)
        
        # if self.is_training:
        #     self.normal = self.cfg.ARGUMENT.train.normal.use
        #     self.normal_mean = self.cfg.ARGUMENT.train.normal.mean
        #     self.normal_std = self.cfg.ARGUMENT.train.normal.std
        #     self.aug_params = self.cfg.ARGUMENT.train
        #     self.flag = 'Train'
        # else:
        #     self.normal = self.cfg.ARGUMENT.val.normal.use
        #     self.normal_mean = self.cfg.ARGUMENT.val.normal.mean
        #     self.normal_std = self.cfg.ARGUMENT.val.normal.std
        #     self.aug_params = self.cfg.ARGUMENT.val
        #     self.flag = 'Val'

        self.aug_params = self.cfg.get('ARGUMENT')[phase]
        self.dataset_params = self.cfg.DATASET
        # set up the keys of the dataset
        self.setup()
        self.custom_setup()

    def setup(self):
        # self.image_loader = ImageLoader(read_format=self.cfg.DATASET.read_format, channel_num=self.cfg.DATASET.channel_num, channel_name=self.cfg.DATASET.channel_name)
        self.image_loader = ImageLoader(read_format=self.dataset_params.read_format, channel_num=self.dataset_params.channel_num, channel_name=self.dataset_params.channel_name)
        # self.video_loader = VideoLoader(self.image_loader, params=self.aug_params, transforms=self.transforms, normalize=self.normal, mean=self.normal_mean, std=self.normal_std)
        self.video_loader = VideoLoader(self.image_loader, params=self.aug_params, transforms=self.transforms, normalize=self.aug_params.normal, mean=self.aug_params.normal.mean, std=self.aug_params.normal.std)
        if self.mini:
            self.video_nums = len(self.videos_keys)
            print(f'The read format of MINI dataset is {self.dataset_params.read_format} in {self._NAME}')
        elif self.one_video:
            print(f'The read format of ONE VIDEO dataset is {self.dataset_params.read_format} in {self._NAME}')
        else:
            print(f'The read format of dataset is {self.dataset_params.read_format} in {self._NAME}')
    
    def custom_setup(self):
        print(f'Not re-implementation of custom setup in {AbstractVideoAnomalyDataset._NAME}')
        # pass
    
        
    def __getitem__(self, indice):
        if self.one_video:
            video_name = list(self.videos_keys)[0]
        elif self.mini:
            temp = indice % self.video_nums
            video_name = list(self.videos_keys)[temp]
        else:
            video_name = list(self.videos_keys)[indice]

       
        item = self._get_frames(video_name)

        annotation = self._get_annotations(video_name)
        
        meta = self._get_meta(video_name)
        
        return item, annotation, meta

    def _get_frames(self, video_name):
        '''
        get the frames 
        '''
        return [] 
    
    def _get_annotations(self, video_name):
        '''
        get the frames
        '''
        return []
    
    def _get_meta(self, video_name):
        '''
        get the meta data 
        '''
        return []


    def __len__(self):
        if self.one_video and self.mini:
            return self.cfg.DATASET.mini_dataset.samples
        
        if self.one_video:
            return self.pics_len
        elif self.mini:
            return self.cfg.DATASET.mini_dataset.samples
        else:
            return self.videos.__len__() # the number of the videos

