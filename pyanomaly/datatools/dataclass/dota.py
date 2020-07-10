import torch
import numpy as np
import cv2
from collections import OrderedDict
import glob
import os
from torch.utils.data import Dataset
from pyanomaly.datatools.abstract.anomaly_video_dataset import AbstractVideoAnomalyDataset
from pyanomaly.datatools.abstract.tools import ImageLoader, VideoLoader

class DOTA(AbstractVideoAnomalyDataset):
    _NAME='DoTA Dataset'
    def custom_setup(self):
        self.image_loader = ImageLoader(read_format=self.cfg.DATASET.read_format, channel_num=self.cfg.DATASET.channel_num, channel_name=self.cfg.DATASET.channel_name)
        self.video_loader = VideoLoader(self.image_loader, params=self.aug_params, transforms=self.transforms, normalize=self.normal, mean=self.normal_mean, std=self.normal_std)
        print(f'The read format of dataset is {self.cfg.DATASET.read_format} in {self._NAME}')
        
    def _get_frames(self, indice):
        video_name = list(self.videos_keys)[indice]
        cusrsor = self.videos[video_name]['cursor']
        if (cusrsor + self.clip_length) > self.videos[video_name]['length']:
            cusrsor = 0
        start = cusrsor

        video_clip, video_clip_original = self.video_loader.read(self.videos[video_name]['frames'], start, start+self.clip_length, clip_length=self.sampled_clip_length, step=self.frame_step)
        self.videos[video_name]['cursor'] = cusrsor + self.clip_step

        return video_clip, video_clip_original


def get_dota(cfg, flag, aug):
    pass
    