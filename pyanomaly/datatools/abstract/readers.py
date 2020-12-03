"""
@author:  Yuhao Cheng
@contact: yuhao.cheng[at]outlook.com
"""
import cv2
import torch
from PIL import Image
import numpy as np
import torchvision.transforms.functional as tf
import os
import numpy as np
import scipy.io as scio
import imgaug.augmenters as iaa
import logging
logger = logging.getLogger(__name__)

__all__ = ['ImageLoader', 'VideoLoader', 'GroundTruthLoader']

class ImageLoader(object):
    _support_format = ['opencv', 'pillow']
    def __init__(self, read_format='pillow', channel_num=3, channel_name='rgb',params=None, transforms=None, normalize=False, mean=None, std=None, deterministic=False):
        '''
        read_format: use which package to read the image, 'opencv' or 'pillow'
        transforms: how to augment the image
        '''
        self.read_format = read_format
        self.channel_num = channel_num
        self.channel_name = channel_name
        self.params = params
        # Convert this augmenter from a stochastic to a deterministic one.
        if deterministic:
            self.transforms = transforms.to_deterministic()
        else:
            self.transforms = transforms
            
        self.normalize = normalize
        self.mean = mean
        self.std = std
        assert read_format in ImageLoader._support_format, f'the read function is not supported, {read_format}'
    
    def read(self, name, flag='other', array_type='tensor'):
        '''
        name: the absolute path of the image
        flag: use the torchvision transforms ----> 'torchvision'
              use other opensource transforms, like imgaug -----> 'other'
        array_type:  the return type of the image. 'tensor' | 'ndarray'
        '''
        image = None
        if self.read_format == 'opencv':
            image = cv2.imread(name)
            image = image[:,:,[2,1,0]] # change to the RGB
            if self.channel_name == 'gray':
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                image = np.expand_dims(image, axis=2)
                # import ipdb; ipdb.set_trace()
        elif self.read_format == 'pillow':
            image = Image.open(name)
            if self.channel_name == 'gray':
                image = image.convert('L')
        
        assert image is not None, f'Not read the image:{name}'
        
        original_image = self._get_original(image)

        if self.transforms is not None:
            image = self._augment(image, flag)
        
        if array_type == 'tensor':
            if type(image).__name__ == 'Tensor':
                return image, original_image
            if 'PIL' in str(type(image)):
                image = tf.to_tensor(image)
                if self.normalize:
                    image = tf.normalize(image, mean=self.mean, std=self.std)
            
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image.transpose((2, 0, 1))) # Conver the channel order to [C, H, W]
                if self.normalize:
                    # Normalize the RGB to [0, 1.0]
                    image = image.div(255.0)
                    if (len(self.mean)!=0) and (len(self.std)!=0):
                        # Based on the mean and std to normalize the image
                        image_channel_num = image.shape[0]
                        if image_channel_num == len(self.mean):
                            image = tf.normalize(image, mean=self.mean, std=self.std)
                        else:
                            raise Exception(f'The number of image channel is {image_channel_num} vs the number of mean is {len(self.mean)} and the number of std is {len(self.std)}') 
        elif array_type == 'ndarray':
            if isinstance(image, np.ndarray):
                return image, original_image
            elif type(image).__name__ == 'Tensor':
                image = image.numpy()
            
        return image, original_image

    def _get_original(self, image):
        if self.read_format == 'opencv':
            image = torch.from_numpy(image)
        elif self.read_format == 'pillow':
            if image.mode == 'I':
                image = torch.from_numpy(np.array(image, np.int32, copy=False))
            elif image.mode == 'I;16':
                image = torch.from_numpy(np.array(image, np.int16, copy=False))
            elif image.mode == 'F':
                image = torch.from_numpy(np.array(image, np.float32, copy=False))
            elif image.mode == '1':
                image = 255 * torch.from_numpy(np.array(image, np.uint8, copy=False))
            else:
                image = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        
        if self.normalize:
            image = image.div(255.0)
            if (len(self.mean)!=0) and (len(self.std)!=0):
                # Based on the mean and std to normalize the image
                image_channel_num = image.shape[0]
                if image_channel_num == len(self.mean):
                    image = tf.normalize(image, mean=self.mean, std=self.std)
                else:
                    raise Exception(f'GET original image: The number of image channel is {image_channel_num} vs the number of mean is {len(self.mean)} and the number of std is {len(self.std)}')
        return image

    def _augment(self, image, flag):
        if type(image).__name__ == 'ndarray' and flag == 'torchvision':
            image = tf.to_pil_image(image) # the ndarray should be RGB not BGR, refer to the source of pytorch

        if type(image).__name__ != 'ndarray' and flag == 'other':
            image = tf.to_tensor(image) # it will normalize the data to [0, 1]
            image = image.numpy()

        image = self.transforms(image)
        
        return image

        
class VideoLoader(object):
    def __init__(self, image_loader, params=None, transforms=None, normalize=False, mean=None, std=None):
        self.params = params
        self.transforms = transforms
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.image_loader = image_loader

    def read(self, frames_list, start, end, clip_length=2, step=1, array_type='tensor'):
        '''
        array_type: the output format of the video array. The shape of the video data is [C,D,H,W]
        '''
        clip_list = list()
        clip_list_original = list()
        for frame_id in range(start, end, step):
            # import ipdb; ipdb.set_trace()
            frame_name = frames_list[frame_id]
            frame, original_frame = self.image_loader.read(frame_name, array_type='ndarray')
            original_frame = original_frame.numpy()
            clip_list.append(frame)
            clip_list_original.append(original_frame)
        
        # Make the clip have the same length, method1: supplement the frames
        if len(clip_list) < clip_length:
            diff = clip_length - len(clip_list)
            # print(f'clip_len:{len(clip_list)}, diff:{diff}')
            clip_list.extend([clip_list[-1]] * diff)
            clip_list_original.extend([clip_list_original[-1]] * diff)

        # method2: drop it
        # =================================        
        # not implement
        # =================================
        clip_np = np.array(clip_list)  # the shape of the clip_np is [D,H,W,C]
        clip_original = self._normalize_original(torch.from_numpy(np.array(clip_list_original)))  # the shape of the clip_original is [C, D, H, W]
        
        assert clip_np.shape[0] == clip_length, f'The clip length is {clip_length}, the real one is {clip_np[0]}'
        
        # Use the data augment for the video
        if self.transforms is not None:
            # type of clip is ndarray
            clip = self._augment(clip_np)
        
        if array_type == 'ndarray':
            if isinstance(clip, np.ndarray):
                return clip, clip_original
            else:
                raise Exception('Some error in videoloader line 121')
        elif array_type == 'tensor':
            if isinstance(clip, torch.Tensor):
                return clip, clip_original
            elif isinstance(clip, np.ndarray):
                clip = torch.from_numpy(clip.transpose(0,3,1,2)) # Conver the channel order to [D,C,H,W]
                clip = clip * 1.0
                # import ipdb; ipdb.set_trace()
                if self.normalize:
                    # Normalize 
                    clip = clip.div(255.0)
                    if (len(self.mean) != 0 ) and (len(self.std) != 0):
                        # Based on the mean and std to normalize the image
                        # import ipdb; ipdb.set_trace()
                        for temp in clip:
                            temp_channel_num = temp.shape[0]
                            if temp_channel_num == len(self.mean):
                                tf.normalize(temp, mean=self.mean, std=self.std, inplace=True)
                            else:
                                raise Exception(f'\033[1;31m The shape of frame  is {temp.shape} vs the number of mean is {len(self.mean)} and the number of std is {len(self.std)}\033[0m')
            else:
                raise Exception('Some error in videoloader line 134')
        else:
            raise Exception(f'Get the wrong type {array_type}')
        
        clip = clip.permute(1,0,2,3)
        return clip, clip_original  # the batch shape is [N,C,D,H,W], because the Pytorch conv3D is [N,C,D,H,W]

    def _augment(self, images):
        '''
        We need the augmentation to keep the temporal information of the clip,
        so all augmentations in one clip without random
        '''
        # Get the resize function in the augmentation
        if self.params.resize.use:
            resize = self.transforms.find_augmenters_by_name('resize')[0]
        elif self.params.CropToFixedSize.use:
            resize = iaa.Resize({"height": self.params.CropToFixedSize.height, "width": self.params.CropToFixedSize.width}, name='resize')
        else:
            raise Exception('YOU MUST HAVE THE SAME SIZE OF IMAGES')
        # Make the some data not augment
        oneof_iaa = iaa.OneOf([self.transforms, resize])
        oneof_iaa_deterministic = oneof_iaa.to_deterministic()
        temp = [oneof_iaa_deterministic(image=images[i]) for i in range(len(images))]
        temp = np.array(temp)
        # import ipdb; ipdb.set_trace()
        return temp
    
    def _normalize_original(self, clip):
        '''
        clip [D, H, W, C]
        '''
        if self.normalize:
            clip = clip.permute(0,3,1,2)
            clip = clip * 1.0 / 255.0
            for temp in clip:
                if (len(self.mean) != 0 ) and (len(self.std) != 0):
                    # Based on the mean and std to normalize the image
                    # import ipdb; ipdb.set_trace()
                    temp_channel_num = temp.shape[0]
                    if temp_channel_num == len(self.mean):
                        tf.normalize(temp, mean=self.mean, std=self.std, inplace=True)
                    else:
                        raise Exception(f'\033[1;31m Normalize Original: The shape of frame  is {temp.shape} vs the number of mean is {len(self.mean)} and the number of std is {len(self.std)}\033[0m')
        else:
            clip = clip.permute(0,3,1,2) # [D,C,H,W]
        
        clip = clip.permute(1,0,2,3)
        return clip


class GroundTruthLoader(object):
    # give the name of the supported datasets
    Avenue = 'Avenue'
    Shanghai = 'Shanghai'
    Ped1 = 'Ped1'
    Ped2 = 'Ped2'

    # _NAME = [Avenue, Shanghai, Ped1, Ped2]

    _LABEL_FILE = {
        Avenue: 'avenue.mat',
        Ped1:'ped1.mat',
        Ped2:'ped2.mat',
    }

    # def __init__(self, cfg, is_training=False):
    #     self.cfg = cfg
    #     # judge the support the dataset
    #     if self.cfg.DATASET.name not in GroundTruthLoader._NAME:
    #         raise Exception('Not support the dataste')
    #     else:
    #         self.name = self.cfg.DATASET.name
        
    #     self.gt_path = self.cfg.DATASET.gt_path
    def __init__(self) -> None:
        self.dataset_name = ''
        self.gt_path = ''
        self.data_path = ''
    
    def set_name(self, dataset_name):
        self.dataset_name = dataset_name
    
    def set_gt_path(self,gt_path):
        self.gt_path = gt_path
    
    def set_data_path(self, data_path):
        self.data_path = data_path

    # def __call__(self):
    #     if self.name == GroundTruthLoader.Shanghai:
    #         gt = self._load_shanghai_gt()
    #     else:
    #         gt = self._load_avenue_ped1_ped2_gt()
        
    #     return gt
    
    def read(self, dataset_name, gt_path, data_path):
        self.set_name(dataset_name)
        self.set_gt_path(gt_path)
        self.set_data_path(data_path)
        logger.info(f'Read the ground truth of dataset {self.dataset_name} in {self.gt_path} of {self.data_path}')
        if dataset_name == GroundTruthLoader.Shanghai:
            gt = self._load_shanghai_gt()
        elif dataset_name in [GroundTruthLoader.Avenue, GroundTruthLoader.Ped1, GroundTruthLoader.Ped2]:
            gt = self._load_avenue_ped1_ped2_gt()
        else:
            raise Exception(f'Not Support dataset: {self.dataset_name}')

        # pass
        return gt

    def _load_avenue_ped1_ped2_gt(self):
        mat_file = os.path.join(self.gt_path, GroundTruthLoader._LABEL_FILE[self.dataset_name])
        abnormal_events = scio.loadmat(mat_file, squeeze_me=True)['gt']
        
        number_videos = abnormal_events.shape[0]

        # dataset_video_folder = self.cfg.DATASET.test_path
        dataset_video_folder = self.data_path
        video_list = sorted(os.listdir(dataset_video_folder))
        
        assert number_videos == len(video_list), f'ground true does not match the number of testing videos. {number_videos} != {len(video_list)}'

        # get the total frames of sub videos
        def get_video_length(sub_video_number):
            video_name = os.path.join(dataset_video_folder, video_list[sub_video_number])
            assert os.path.isdir(video_name), f'{video_name} is not directory!'

            length = len(os.listdir(video_name))

            return length
        
        gt = []
        for i in range(number_videos):
            length = get_video_length(i)

            sub_video_gt = np.zeros((length,), dtype=np.int8)
            sub_abnormal_events = abnormal_events[i]

            if sub_abnormal_events.ndim == 1:
                sub_abnormal_events = sub_abnormal_events.reshape((sub_abnormal_events.shape[0], -1))
            
            _, number_abnormal = sub_abnormal_events.shape
        
            for j in range(number_abnormal):
                # (start - 1, end - 1)
                start = sub_abnormal_events[0, j] - 1 # the first row is the start point
                end = sub_abnormal_events[1, j]   # the second row is the end point

                sub_video_gt[start:end] = 1
        
            gt.append(sub_video_gt)
        # import ipdb; ipdb.set_trace()
        return gt

    def _load_shanghai_gt(self):
        video_path_list = sorted(os.listdir(self.gt_path))

        gt = []
        for video in video_path_list:
            gt.append(np.load(os.path.join(self.gt_path, video)))
        return gt

