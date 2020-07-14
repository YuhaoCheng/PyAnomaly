import glob
import os
from collections import OrderedDict
from torch.utils.data import Dataset

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
            self.flag = 'Train'
        else:
            self.flag = 'Not Train'
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
            print(f'\033[1;34m The clip number of {self.cfg.DATASET.name}#{self.flag}is:{self.total_clips} \033[0m')
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
            print(f'\033[1;34m The clip number of one video {video_name}#{self.flag} is:{self.total_clips_onevideo} of {self.cfg.DATASET.name}\033[0m') 
    
    def __getitem__(self, indice):
        raise Exception(f'No inplement at {AbstractVideoDataset._NAME}')
    
    def __len__(self):
        raise Exception(f'No implement at {AbstractVideoDataset._NAME}')
    