import torch
import glob
import os
from collections import OrderedDict
from .image_dataset import AbstractImageDataset
# from .tools import np_load_frame

class AbstractVideoAnomalyDataset(AbstractImageDataset):
    _name = 'AbstractVideoAnomalyDataset'
    # def __init__(self, dataset_folder, clip_length, size=(256, 256), transforms=None, is_training=True, only_frame=True, extra=False, **kwargs):
    def __init__(self, dataset_folder, clip_length, frame_step=1, clip_step=1,transforms=None, is_training=True, one_video=False, only_frame=True, extra=False, **kwargs):
        '''
        size = (h, w)
        is_training: True-> only get the frames, False-> get the frame and annotations
        '''
        super(AbstractVideoAnomalyDataset, self).__init__(dataset_folder, transforms)
        self.videos = OrderedDict()
        self.cfg = kwargs['cfg']
        self.clip_length = clip_length
        self.frame_step = frame_step
        self.clip_step = clip_step
        self.is_training = is_training
        self.one_video = one_video
        self.only_frame = only_frame
        self.extra = extra
        self.kwargs = kwargs

        if self.is_training:
            self.normal = self.cfg.ARGUMENT.train.normal.use
            self.normal_mean = self.cfg.ARGUMENT.train.normal.mean
            self.normal_std = self.cfg.ARGUMENT.train.normal.std
            self.aug_params = self.cfg.ARGUMENT.train
        else:
            self.normal = self.cfg.ARGUMENT.val.normal.use
            self.normal_mean = self.cfg.ARGUMENT.val.normal.mean
            self.normal_std = self.cfg.ARGUMENT.val.normal.std
            self.aug_params = self.cfg.ARGUMENT.val
        # set up the keys of the dataset
        self.setup()
        self.custom_setup()

    def custom_setup(self):
        print(f'The kwargs in custom_setup of {AbstractVideoAnomalyDataset._name} are:{self.kwargs}')
        pass
    
    def setup(self):
        if not self.one_video:
            # the dir is the path of the whole dataset
            videos = glob.glob(os.path.join(self.dir, '*'))
            self.total_clips = 0
            for video in sorted(videos):
                video_name = video.split('/')[-1]
                # print(video)
                self.videos[video_name] = OrderedDict()
                self.videos[video_name]['path'] = video
                self.videos[video_name]['frames'] = glob.glob(os.path.join(video, '*.jpg'))
                self.videos[video_name]['frames'].sort()
                self.videos[video_name]['length'] = len(self.videos[video_name]['frames'])
                self.videos[video_name]['cursor'] = 0
                self.total_clips += (len(self.videos[video_name]['frames']) - self.clip_length)
            self.videos_keys = self.videos.keys()
            print(f'\033[1;34m The clips are:{self.total_clips} \033[0m')
            # self.cursor = 0
        else:
            # the dir is the path of one video
            video_name = os.path.split(self.dir)[-1]
            self.videos['name'] = video_name
            self.videos['path'] = self.dir
            self.videos['frames'] =glob.glob(os.path.join(self.dir,'*.jpg'))
            self.videos['frames'].sort()
            self.pics_len=len(self.videos['frames'])

    def __getitem__(self, indice):
        # item, meta_data = self._get_frames(indice)
        item = self._get_frames(indice)

        # only get the frame, or in general, the item
        if self.only_frame:
            return item 

        if not self.is_training:
            annotation = self._get_annotations(indice)
        else:
            annotation = 'None'
        if self.extra:
            custom = self._custom_get(indice)
        else:
            custom = 'None'
        return item, annotation, custom

    def _get_frames(self, indice):
        '''
        get the frames 
        '''
        pass

    def _get_annotations(self, indice):
        pass

    def _custom_get(self, indice):
        pass

    def __len__(self):
        return self.videos.__len__() # the number of the videos

