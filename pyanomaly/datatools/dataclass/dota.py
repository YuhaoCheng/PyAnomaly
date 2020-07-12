import torch
import numpy as np
import cv2
import json
import glob
import os
from collections import OrderedDict
from torch.utils.data import Dataset
from pyanomaly.datatools.abstract.anomaly_video_dataset import AbstractVideoAnomalyDataset
from pyanomaly.datatools.abstract.tools import ImageLoader, VideoLoader

class DOTA(AbstractVideoAnomalyDataset):
    _NAME='DoTA Dataset'

    def _get_frames(self, video_name):
        # video_name = list(self.videos_keys)[indice]
        cusrsor = self.videos[video_name]['cursor']
        if (cusrsor + self.clip_length) > self.videos[video_name]['length']:
            cusrsor = 0
        start = cusrsor
        self._current_start = cusrsor
        video_clip, video_clip_original = self.video_loader.read(self.videos[video_name]['frames'], start, start+self.clip_length, clip_length=self.sampled_clip_length, 
                                                                 step=self.frame_step)
        self.videos[video_name]['cursor'] = cusrsor + self.clip_step

        return video_clip, video_clip_original
    
    def custom_setup(self):
        anno_path = self.cfg.DATASET.gt_path
        anno_files = os.listdir(anno_path)
        if self.one_video:
            anno_files = [f'{self.video_keys[0]}.json']
            assert len(self.video_keys) == 1, f'The one video dataset must have one vide, in fact is has {len(self.video_keys)}'
        else:
            assert len(anno_files) == len(self.videos.videos_keys), f'The number of videos and annos are different, the video:{len(self.video_keys)} and the annos:{len(anno_files)}'
        
        anno_files = sorted(anno_files)
        for json_file in anno_files:
            with open(os.path.join(self.cfg.DATASET.gt_path, json_file), 'r') as f:
                json_data = json.load(f)
                anno_video_name = json_data['video_name']
                self.annos[anno_video_name] = OrderedDict()
                self.annos[anno_video_name]['accident_id'] = json_data['accident_id']
                self.annos[anno_video_name]['labels'] = json_data['labels']
    
    def _get_annotations(self, video_name):
        # video_name = list(self.videos_keys)[indice]
        anno = self.annos[video_name]
        labels = anno['labels']
        accident_id = self.annos[video_name]['accident_id']
        temp = list()
        for index in range(self._current_start, self._current_start+self.clip_length, self.frame_step):
            label = labels[index]
            assert index == label['frame_id'], 'Some Wrong!! in reading the annos'
            if label['accident_name'] == "normal":
                temp.append(0)
            else:
                temp.append(accident_id)
        
        



def get_dota(cfg, flag, aug):
    pass
    