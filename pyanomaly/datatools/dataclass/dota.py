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
        # self.image_loader = ImageLoader(read_format=self.cfg.DATASET.read_format,transforms=self.transforms, normalize=self.normal, mean=self.normal_mean, std=self.normal_std)
        self.video_loader = VideoLoader(self.image_loader, params=self.aug_params, transforms=self.transforms, normalize=self.normal, mean=self.normal_mean, std=self.normal_std)
        print(f'The read format of dataset is {self.cfg.DATASET.read_format} in {self._NAME}')
    