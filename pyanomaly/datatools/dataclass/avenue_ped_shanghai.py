import torch
import numpy as np
import cv2
from collections import OrderedDict
import glob
import os
from torch.utils.data import Dataset
from pyanomaly.datatools.abstract.anomaly_video_dataset import AbstractVideoAnomalyDataset
from pyanomaly.datatools.abstract.tools import ImageLoader, VideoLoader

class AvenuePedShanghai(AbstractVideoAnomalyDataset):
    _NAME = 'AvenuePedShanghai Dataset'
    def custom_setup(self):
        self.image_loader = ImageLoader(read_format=self.cfg.DATASET.read_format, channel_num=self.cfg.DATASET.channel_num, channel_name=self.cfg.DATASET.channel_name)
        # self.image_loader = ImageLoader(read_format=self.cfg.DATASET.read_format,transforms=self.transforms, normalize=self.normal, mean=self.normal_mean, std=self.normal_std)
        self.video_loader = VideoLoader(self.image_loader, params=self.aug_params, transforms=self.transforms, normalize=self.normal, mean=self.normal_mean, std=self.normal_std)
        print(f'The read format of train dataset is {self.cfg.DATASET.read_format} in {self._NAME}')
    
    def _get_frames(self, indice):
        key = list(self.videos_keys)[indice]
        cusrsor = self.videos[key]['cursor']
        if (cusrsor + self.clip_length) > self.videos[key]['length']:
            cusrsor = 0
        start = cusrsor

        video_clip, video_clip_original = self.video_loader.read(self.videos[key]['frames'], start, start+self.clip_length, clip_length=self.sampled_clip_length,step=self.frame_step)
        self.videos[key]['cursor'] = cusrsor + self.clip_step
        return video_clip, video_clip_original
    
    def get_image(self, image_name):
        # keep for debug
        image =  self.image_loader.read(image_name)
        return image
    

class AvenuePedShanghaiOneVideo(AbstractVideoAnomalyDataset):
    '''
    The only get the one video, not whole dataset.
    So we wil use it for each video in whole video
    '''
    _NAME = 'AvenuePedShanghaiOneVideo Dataset'
    def __init__(self, dataset_folder, clip_length, sampled_clip_length,frame_step, clip_step=1, transforms=None, is_training=True, one_video=True, cfg=None):
        super(AvenuePedShanghaiOneVideo, self).__init__(dataset_folder, clip_length, sampled_clip_length=sampled_clip_length, frame_step=frame_step,clip_step=clip_step, transforms=transforms, is_training=is_training, one_video=one_video, cfg=cfg)

    def custom_setup(self):
        self.image_loader = ImageLoader(read_format=self.cfg.DATASET.read_format, channel_num=self.cfg.DATASET.channel_num, channel_name=self.cfg.DATASET.channel_name)
        # self.image_loader = ImageLoader(read_format=self.cfg.DATASET.read_format,transforms=self.transforms, normalize=self.normal, mean=self.normal_mean, std=self.normal_std)
        self.video_loader = VideoLoader(self.image_loader, params=self.aug_params, transforms=self.transforms, normalize=self.normal, mean=self.normal_mean, std=self.normal_std)

    def __len__(self):
        return self.pics_len

    def _get_frames(self, indice):
        video_name = list(self.videos_keys)[0]
        start = (indice * self.clip_step) % self.pics_len
        if start + self.clip_length >= self.pics_len:
            end = self.pics_len - 1
        else:
            end = start + self.clip_length
        video_clip, video_clip_original = self.video_loader.read(self.videos[video_name]['frames'], start, end, clip_length=self.sampled_clip_length, step=self.frame_step)
        return video_clip, video_clip_original
    
    def get_image(self,name):
        # keep for debug
        image = self.image_loader.read(name)
        return image

class MiniAvenuePedShanghai(AbstractVideoAnomalyDataset):
    _NAME = 'MiniAvenuePedShanghai Dataset'
    def custom_setup(self):
        # self.image_loader = ImageLoader(read_format=self.cfg.DATASET.read_format,transforms=self.transforms, normalize=self.normal, mean=self.normal_mean, std=self.normal_std)
        self.image_loader = ImageLoader(read_format=self.cfg.DATASET.read_format, channel_num=self.cfg.DATASET.channel_num, channel_name=self.cfg.DATASET.channel_name)
        self.video_loader = VideoLoader(self.image_loader, params=self.aug_params, transforms=self.transforms, normalize=self.normal, mean=self.normal_mean, std=self.normal_std)
        self.video_nums = len(self.videos_keys)

    def _get_frames(self, indice):
        temp = indice % self.video_nums
        key = list(self.videos_keys)[temp]
        rng = np.random.RandomState(2020)
        # import ipdb; ipdb.set_trace()
        start = rng.randint(0, self.videos[key]['length'] - self.clip_length)
        video_clip, video_clip_original = self.video_loader.read(self.videos[key]['frames'], start, start+self.clip_length, clip_length=self.sampled_clip_length, step=self.frame_step)
        return video_clip, video_clip_original

    def __len__(self):
        return self.cfg.DATASET.mini_dataset.samples
    
    def get_image(self, image_name):
        image =  self.image_loader.read(image_name)
        return image


# -----------------Functions Part-------------------------
def _get_test_dataset(cfg, aug):
    dataset_list = OrderedDict()    
    video_dirs = os.listdir(cfg.DATASET.test_path)
    video_dirs.sort()
    for t_dir in video_dirs:
        _temp_test_folder = os.path.join(cfg.DATASET.test_path, t_dir)
        dataset = AvenuePedShanghaiOneVideo(_temp_test_folder, clip_length=cfg.DATASET.test_clip_length, sampled_clip_length=cfg.DATASET.test_sampled_clip_length,clip_step=cfg.DATASET.test_clip_step, frame_step=cfg.DATASET.test_frame_step,transforms=aug, cfg=cfg)
        dataset_list[t_dir] = dataset
    video_keys = list(dataset_list.keys())
    return (dataset_list, video_keys)

def _get_train_w_dataset(cfg, aug):
    dataset_list = OrderedDict()
    video_dirs = os.listdir(cfg.DATASET.train_path)
    video_dirs.sort()
    for t_dir in video_dirs:
        _temp_test_folder = os.path.join(cfg.DATASET.train_path, t_dir)
        if cfg.DATASET.name == 'shanghai':
            dataset = MiniAvenuePedShanghai(_temp_test_folder, clip_length=cfg.DATASET.test_clip_length, sampled_clip_length=cfg.DATASET.test_sampled_clip_length, clip_step=cfg.DATASET.test_clip_step, frame_step=cfg.DATASET.test_frame_step, one_video=True, transforms=aug, cfg=cfg)
            print(f'\033[1;31mUsing the MINI dataset of {cfg.DATASET.name} for calc the w \033[0m')
        else:
            dataset = AvenuePedShanghaiOneVideo(_temp_test_folder, clip_length=cfg.DATASET.train_clip_length, sampled_clip_length=cfg.DATASET.train_sampled_clip_length, clip_step=cfg.DATASET.train_clip_step, frame_step=cfg.DATASET.train_frame_step,transforms=aug, cfg=cfg)
        dataset_list[t_dir] = dataset
    video_keys = list(dataset_list.keys())
    return (dataset_list, video_keys)

def _get_cluster_dataset(cfg, aug):
    dataset_list = OrderedDict()
    
    video_dirs = os.listdir(cfg.DATASET.train_path)
    video_dirs.sort()
    for t_dir in video_dirs:
        _temp_train_folder = os.path.join(cfg.DATASET.train_path, t_dir)
        dataset = AvenuePedShanghaiOneVideo(_temp_train_folder, clip_length=cfg.DATASET.train_clip_length, sampled_clip_length=cfg.DATASET.train_sampled_clip_length, clip_step=cfg.DATASET.train_clip_step, frame_step=cfg.DATASET.train_frame_step,is_training=True, transforms=aug, cfg=cfg)
        dataset_list[t_dir] = dataset
    video_keys = list(dataset_list.keys())
    return (dataset_list, video_keys)

def get_avenue_ped_shanghai(cfg, flag, aug):
    '''
    Using the function to register the Dataset
    '''
    if flag == 'train':
        t = AvenuePedShanghai(cfg.DATASET.train_path, clip_length=cfg.DATASET.train_clip_length, sampled_clip_length=cfg.DATASET.train_sampled_clip_length, clip_step=cfg.DATASET.train_clip_step, frame_step=cfg.DATASET.train_frame_step,transforms=aug, cfg=cfg)
    elif flag == 'val':
        t = MiniAvenuePedShanghai(cfg.DATASET.test_path, clip_length=cfg.DATASET.test_clip_length, sampled_clip_length=cfg.DATASET.test_sampled_clip_length, clip_step=cfg.DATASET.test_clip_step, frame_step=cfg.DATASET.test_frame_step, transforms=aug, cfg=cfg)
        print(f'\033[1;31mUsing the MINI dataset of {cfg.DATASET.name} for evaluation in the process of training\033[0m')
    elif flag == 'test':
        t = _get_test_dataset(cfg, aug)
    elif flag == 'train_w':
        t = _get_train_w_dataset(cfg, aug)
    elif flag == 'cluster_train':
        t = _get_cluster_dataset(cfg, aug)
    return t 


if __name__ == '__main__':
    import ipdb; ipdb.set_trace()