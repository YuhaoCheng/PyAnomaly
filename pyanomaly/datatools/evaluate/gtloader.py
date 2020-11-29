'''
Will be depreacted in ver2.0
'''
# import os
# import numpy as np
# import scipy.io as scio

# class RecordResult(object):
#     def __init__(self, fpr=None, tpr=None, thresholds=None, auc=-np.inf, dataset=None, loss_file=None, sigma=0):
#         self.fpr = fpr
#         self.tpr = tpr
#         self.thresholds = thresholds
#         self.auc = auc
#         self.dataset = dataset
#         self.loss_file = loss_file
#         self.sigma = sigma

#     def __lt__(self, other):
#         return self.auc < other.auc

#     def __gt__(self, other):
#         return self.auc > other.auc

#     def __str__(self):
#         return f'dataset = {self.dataset}, loss file = {self.loss_file}, auc = {self.auc:.5f}, sigma={self.sigma}'
    
#     def get_threshold(self):
#         diff_list = list()
#         num_points = len(self.fpr)
#         for i in range(num_points):
#             temp = self.tpr[i] - self.fpr[i]
#             diff_list.append(temp)
#         index = diff_list.index(max(diff_list))

#         return self.thresholds[index]


# class GroundTruthLoader(object):
#     # give the name of the supported datasets
#     Avenue = 'avenue'
#     Shanghai = 'shanghai'
#     Ped1 = 'ped1'
#     Ped2 = 'ped2'

#     _NAME = [Avenue, Shanghai, Ped1, Ped2]

#     _LABEL_FILE = {
#         Avenue: 'avenue.mat',
#         Ped1:'ped1.mat',
#         Ped2:'ped2.mat',
#     }

#     def __init__(self, cfg, is_training=False):
#         self.cfg = cfg
#         # judge the support the dataset
#         if self.cfg.DATASET.name not in GroundTruthLoader._NAME:
#             raise Exception('Not support the dataste')
#         else:
#             self.name = self.cfg.DATASET.name
        
#         self.gt_path = self.cfg.DATASET.gt_path
    
#     def __call__(self):
#         if self.name == GroundTruthLoader.Shanghai:
#             gt = self._load_shanghai_gt()
#         else:
#             gt = self._load_avenue_ped1_ped2_gt()
        
#         return gt
    
#     def _load_avenue_ped1_ped2_gt(self):
#         mat_file = os.path.join(self.gt_path, GroundTruthLoader._LABEL_FILE[self.name])
#         abnormal_events = scio.loadmat(mat_file, squeeze_me=True)['gt']
        
#         number_videos = abnormal_events.shape[0]

#         dataset_video_folder = self.cfg.DATASET.test_path
#         video_list = sorted(os.listdir(dataset_video_folder))
        
#         assert number_videos == len(video_list), f'ground true does not match the number of testing videos. {number_videos} != {len(video_list)}'

#         # get the total frames of sub videos
#         def get_video_length(sub_video_number):
#             video_name = os.path.join(dataset_video_folder, video_list[sub_video_number])
#             assert os.path.isdir(video_name), f'{video_name} is not directory!'

#             length = len(os.listdir(video_name))

#             return length
        
#         gt = []
#         for i in range(number_videos):
#             length = get_video_length(i)

#             sub_video_gt = np.zeros((length,), dtype=np.int8)
#             sub_abnormal_events = abnormal_events[i]

#             if sub_abnormal_events.ndim == 1:
#                 sub_abnormal_events = sub_abnormal_events.reshape((sub_abnormal_events.shape[0], -1))
            
#             _, number_abnormal = sub_abnormal_events.shape
        
#             for j in range(number_abnormal):
#                 # (start - 1, end - 1)
#                 start = sub_abnormal_events[0, j] - 1 # the first row is the start point
#                 end = sub_abnormal_events[1, j]   # the second row is the end point

#                 sub_video_gt[start:end] = 1
        
#             gt.append(sub_video_gt)
#         # import ipdb; ipdb.set_trace()
#         return gt

#     def _load_shanghai_gt(self):
#         video_path_list = sorted(os.listdir(self.gt_path))

#         gt = []
#         for video in video_path_list:
#             gt.append(np.load(os.path.join(self.gt_path, video)))
#         return gt

