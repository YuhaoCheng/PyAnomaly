import torch
import glob
import os
from collections import OrderedDict
from .image_dataset import AbstractVideoDataset
from pyanomaly.datatools.abstract.tools import ImageLoader, VideoLoader

class AbstractVideoAnomalyDataset(AbstractVideoDataset):
    _NAME = 'AbstractVideoAnomalyDataset'
    def __init__(self, frames_folder, clip_length, sampled_clip_length, frame_step=1, clip_step=1, video_format='.mp4', fps=10, transforms=None, is_training=True, one_video=False, only_frame=True, mini=False, extra=False, cfg=None, **kwargs):
        '''
        size = (h, w)
        is_training: True-> only get the frames, False-> get the frame and annotations
        '''
        super(AbstractVideoAnomalyDataset, self).__init__(self, frames_folder, clip_length, sampled_clip_length, frame_step=frame_step, clip_step=clip_step, video_format=video_format, fps=fps, 
                                                          transforms=transforms, is_training=is_training, one_video=one_video, mini=mini, cfg=cfg, **kwargs)
        # self.videos = OrderedDict()
        # self.cfg = kwargs['cfg']
        # self.clip_length = clip_length
        # self.sampled_clip_length = sampled_clip_length
        # self.frame_step = frame_step
        # self.clip_step = clip_step
        # self.is_training = is_training
        # self.one_video = one_video
        # self.mini = mini
        # self.only_frame = only_frame
        # self.extra = extra
        # self.kwargs = kwargs

        if self.is_training:
            self.normal = self.cfg.ARGUMENT.train.normal.use
            self.normal_mean = self.cfg.ARGUMENT.train.normal.mean
            self.normal_std = self.cfg.ARGUMENT.train.normal.std
            self.aug_params = self.cfg.ARGUMENT.train
            self.flag = 'Train'
        else:
            self.normal = self.cfg.ARGUMENT.val.normal.use
            self.normal_mean = self.cfg.ARGUMENT.val.normal.mean
            self.normal_std = self.cfg.ARGUMENT.val.normal.std
            self.aug_params = self.cfg.ARGUMENT.val
            self.flag = 'Val'
        
        # set up the keys of the dataset
        self.setup()
        self.custom_setup()

    def setup(self):
        self.image_loader = ImageLoader(read_format=self.cfg.DATASET.read_format, channel_num=self.cfg.DATASET.channel_num, channel_name=self.cfg.DATASET.channel_name)
        self.video_loader = VideoLoader(self.image_loader, params=self.aug_params, transforms=self.transforms, normalize=self.normal, mean=self.normal_mean, std=self.normal_std)
        if self.mini:
            self.video_nums = len(self.videos_keys)
            print(f'The read format of MINI dataset is {self.cfg.DATASET.read_format} in {self._NAME}')
        elif self.one_video:
            print(f'The read format of ONE VIDEO dataset is {self.cfg.DATASET.read_format} in {self._NAME}')
        else:
            print(f'The read format of dataset is {self.cfg.DATASET.read_format} in {self._NAME}')
    
    def custom_setup(self):
        print(f'Not re-implementation of custom setup in {AbstractVideoAnomalyDataset._NAME}')
        # pass
    
    # def setup(self):
    #     if not self.one_video:
    #         # the dir is the path of the whole dataset
    #         videos = glob.glob(os.path.join(self.dir, '*'))
    #         self.total_clips = 0
    #         for video in sorted(videos):
    #             video_name = video.split('/')[-1]
    #             # print(video)
    #             self.videos[video_name] = OrderedDict()
    #             self.videos[video_name]['path'] = video
    #             self.videos[video_name]['frames'] = glob.glob(os.path.join(video, f'*.{self.cfg.DATASET.image_format}'))
    #             self.videos[video_name]['frames'].sort()
    #             self.videos[video_name]['length'] = len(self.videos[video_name]['frames'])
    #             self.videos[video_name]['cursor'] = 0
    #             self.total_clips += (len(self.videos[video_name]['frames']) - self.clip_length)
    #         self.videos_keys = self.videos.keys()
    #         print(f'\033[1;34m The clip number of {self.cfg.DATASET.name}#{self.flag}is:{self.total_clips} \033[0m')
    #         # self.cursor = 0
    #     else:
    #         self.total_clips_onevideo = 0
    #         # the dir is the path of one video
    #         video_name = os.path.split(self.dir)[-1]
    #         self.videos[video_name] = OrderedDict()
    #         self.videos[video_name]['name'] = video_name
    #         self.videos[video_name]['path'] = self.dir
    #         self.videos[video_name]['frames'] =glob.glob(os.path.join(self.dir,f'*.{self.cfg.DATASET.image_format}'))
    #         self.videos[video_name]['frames'].sort()
    #         self.videos[video_name]['length'] = len(self.videos[video_name]['frames'])
    #         self.videos[video_name]['cursor'] = 0
    #         self.total_clips_onevideo += (len(self.videos[video_name]['frames']) - self.clip_length)
    #         self.pics_len = len(self.videos[video_name]['frames'])
    #         self.videos_keys = self.videos.keys()
    #         print(f'\033[1;34m The clip number of one video {video_name}#{self.flag} is:{self.total_clips_onevideo} of {self.cfg.DATASET.name}\033[0m')
        
    def __getitem__(self, indice):
        # item, meta_data = self._get_frames(indice)
        if self.one_video:
            video_name = list(self.videos_keys)[0]
        elif self.mini:
            temp = indice % self.video_nums
            video_name = list(self.videos_keys)[temp]
        else:
            video_name = list(self.videos_keys)[indice]

        # video_name = list(self.videos_keys)[indice]
        
        # item = self._get_frames(indice)
        item = self._get_frames(video_name)

        # only get the frame, or in general, the item
        # if self.only_frame:
        #     return item 

        # if not self.is_training:
        #     annotation = self._get_annotations(indice)
        # else:
        #     annotation = 'None'
        # annotation = self._get_annotations(indice)
        annotation = self._get_annotations(video_name)
        # if self.extra:
        #     custom = self._custom_get(indice)
        # else:
        #     custom = 'None'
        # meta = self._get_meta(indice)
        meta = self._get_meta(video_name)
        
        return item, annotation, meta

    def _get_frames(self, video_name):
        '''
        get the frames 
        '''
        return None 
    
    def _get_annotations(self, video_name):
        '''
        get the frames
        '''
        return None
    
    def _get_meta(self, video_name):
        '''
        get the meta data 
        '''
        return None


    def __len__(self):
        if self.one_video and self.mini:
            return self.cfg.DATASET.mini_dataset.samples
        
        if self.one_video:
            return self.pics_len
        elif self.mini:
            return self.cfg.DATASET.mini_dataset.samples
        else:
            return self.videos.__len__() # the number of the videos

