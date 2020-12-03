"""
@author:  Yuhao Cheng
@contact: yuhao.cheng[at]outlook.com
"""
import abc
import glob
import os
from collections import OrderedDict
from torch.utils.data import Dataset
from .readers import ImageLoader, VideoLoader
from ..datatools_registry import DATASET_REGISTRY

__all__ = ['AbstractVideoDataset', 'FrameLevelVideoDataset']
class AbstractVideoDataset(Dataset):
    _NAME = 'AbstractVideoDataset'
    def __init__(self, frames_folder, clip_length, sampled_clip_length, frame_step=1, clip_step=1, video_format='mp4', fps=10, transforms=None, is_training=True, one_video=False, mini=False, cfg=None, **kwargs):
        '''
        Args:
            dataset_folder: the frames folder of one video (video_name/...jpg) or whole datasets (xxx/video_name/xxx.jpg)
            clip_length: the lenght of a clip from a video
            sampled_clip_length: the real length of the clip used to train. Sometimes, as a result of the frame_step != 1, clip_length != sampled_clip_length
            frame_step: the time interval between two consecutive frames
            clip_step:  the time interval between two consecutive clips
            transforms: the data augmentation of clip
            is_training: True-> get the training frames and labels; False-> get the testing or val data and labels
            one_video: default: False
            mini: default: False
            cfg: the configuration of the dataset
        '''
        # the meta structure of video dataset
        self.videos = OrderedDict() # contains 'video_name' ->'path', 'frames', 'length', 'cursor'
        self.annos = OrderedDict()
        self.metas = OrderedDict()

        # the params of dataset
        self.dir = frames_folder
        self.clip_length = clip_length
        self.sampled_clip_length = sampled_clip_length
        self.frame_step = frame_step
        self.clip_step = clip_step
        self.video_format = video_format
        self.fps = fps
        self.transforms = transforms
        self.is_training = is_training
        self.one_video = one_video
        self.mini = mini
        self.cfg = cfg
        self.kwargs = kwargs
        
        if self.is_training:
            # self.flag = 'Train'
            self.flag = 'train'
            self.phase = 'train'
        else:
            # self.flag = 'Not Train'
            self.flag = 'val'
            self.phase = 'val'
        
        self.abstract_setup()


    def abstract_setup(self):
        if not self.one_video:
            # the dir is the path of the whole dataset
            videos = glob.glob(os.path.join(self.dir, '*'))
            self.total_clips = 0
            for video in sorted(videos):
                video_name = video.split('/')[-1]
                self.videos[video_name] = OrderedDict()
                self.videos[video_name]['path'] = video
                self.videos[video_name]['frames'] = glob.glob(os.path.join(video, f'*.{self.cfg.DATASET.image_format}'))
                self.videos[video_name]['frames'].sort()
                self.videos[video_name]['length'] = len(self.videos[video_name]['frames'])
                self.videos[video_name]['cursor'] = 0
                self.total_clips += (len(self.videos[video_name]['frames']) - self.clip_length)
            self.videos_keys = self.videos.keys()
            # print(f'\033[1;34m The clip number of {self.cfg.DATASET.name}#{self.flag}is:{self.total_clips} \033[0m')
        else:
            self.total_clips_onevideo = 0
            # the dir is the path of one video
            video_name = os.path.split(self.dir)[-1]
            self.videos[video_name] = OrderedDict()
            self.videos[video_name]['name'] = video_name
            self.videos[video_name]['path'] = self.dir
            self.videos[video_name]['frames'] =glob.glob(os.path.join(self.dir,f'*.{self.cfg.DATASET.image_format}'))
            self.videos[video_name]['frames'].sort()
            self.videos[video_name]['length'] = len(self.videos[video_name]['frames'])
            self.videos[video_name]['cursor'] = 0
            self.total_clips_onevideo += (len(self.videos[video_name]['frames']) - self.clip_length)
            self.pics_len = len(self.videos[video_name]['frames'])
            self.videos_keys = self.videos.keys()
            # print(f'\033[1;34m The clip number of one video {video_name}#{self.flag} is:{self.total_clips_onevideo} of {self.cfg.DATASET.name}\033[0m') 
    
    def __getitem__(self, indice):
        raise Exception(f'No inplement at {AbstractVideoDataset._NAME}')
    
    def __len__(self):
        raise Exception(f'No implement at {AbstractVideoDataset._NAME}')

@DATASET_REGISTRY.register()
class FrameLevelVideoDataset(AbstractVideoDataset):
    _NAME = 'FrameLevelVideoDataset'
    def __init__(self, frames_folder, clip_length, sampled_clip_length, frame_step=1, clip_step=1, video_format='.mp4', fps=10, transforms=None, is_training=True, one_video=False, only_frame=True, mini=False, extra=False, cfg=None, **kwargs):
        '''
        size = (h, w)
        is_training: True-> only get the frames, False-> get the frame and annotations
        '''
        super(FrameLevelVideoDataset, self).__init__(frames_folder, clip_length, sampled_clip_length, frame_step=frame_step, clip_step=clip_step, video_format=video_format, fps=fps, 
                                                          transforms=transforms, is_training=is_training, one_video=one_video, mini=mini, cfg=cfg, **kwargs)
        
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

        self.aug_params = self.cfg.get('ARGUMENT')[self.phase]
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
        # elif self.one_video:
        #     print(f'The read format of ONE VIDEO dataset is {self.dataset_params.read_format} in {self._NAME}')
        # else:
        #     print(f'The read format of dataset is {self.dataset_params.read_format} in {self._NAME}')
    
    @abc.abstractmethod
    def custom_setup(self):
        # print(f'Not re-implementation of custom setup in {FrameLevelVideoDataset._NAME}')
        pass
    
        
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

    # def _get_frames(self, video_name):
    #     '''
    #     get the frames 
    #     '''
    #     return []
    
    def _get_frames(self, video_name):
        cusrsor = self.videos[video_name]['cursor']
        if (cusrsor + self.clip_length) > self.videos[video_name]['length']:
            cusrsor = 0
        if self.mini:
            rng = np.random.RandomState(2020)
            start = rng.randint(0, self.videos[video_name]['length'] - self.clip_length)
        else:
            start = cusrsor

        video_clip, video_clip_original = self.video_loader.read(self.videos[video_name]['frames'], start, start+self.clip_length, clip_length=self.sampled_clip_length, 
                                                                 step=self.frame_step)
        self.videos[video_name]['cursor'] = cusrsor + self.clip_step
        return video_clip
    
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


