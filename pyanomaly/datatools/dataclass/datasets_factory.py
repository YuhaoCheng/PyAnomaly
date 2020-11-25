from os import terminal_size
from ..datasets_registry import DATASET_FACTORY_REGISTRY
from ..datasets_registry import DATASET_REGISTRY
from .avenue_ped_shanghai import *
from ..abstract.abstract_dataset_factory import AbstractDatasetFactory, GetWDataset, GetClusterDataset
from collections import OrderedDict, namedtuple
import os
__all__ = ['VideoAnomalyDatasetFactory']
# @DATASET_FACTORY_REGISTRY.registry()
# class AvenueFactory(object):
#     NORMAL = ['stae', 'amc']
#     CLASS1 = ['ocae']
#     CLASS2 = ['memae']
#     def __init__(self, cfg, aug) -> None:
#         self.cfg = cfg
#         self.model_name = cfg.MODEL.name
#         self.ingredient = DATASET_REGISTRY.get('AvenuePedShanghaiAll')
#         self.aug = aug

#     def _build_normal(self):
#         """
#         """
#         return 0
#     def _build_class1(self):
#         """
#         """
#         return 0

#     def _build_class2(self):
#         """
#         """
#         return 0

#     def _build(self):
#         if self.model_name in AvenueFactory.NORMAL:
#             dataloader = self._build_normal()
#         elif self.model_name in AvenueFactory.CLASS1:
#             dataloader = self._build_class1()
#         elif self.model_name in AvenueFactory.CLASS2:
#             dataloader = self._build_class2()
#         else:
#             raise Exception('123')
        
#         return dataloader
    
#     def __call__(self):
#         dataset_dict = self._build()
#         return dataset_dict

@DATASET_FACTORY_REGISTRY.register()
class VideoAnomalyDatasetFactory(AbstractDatasetFactory, GetWDataset, GetClusterDataset):
    # NORMAL = ['stae', 'amc', 'anopcn', 'anopred']
    NORMAL = ['amc', 'anopcn', 'anopred']
    NEED_W = ['memae', 'stae']
    # NEED_W = ['memae']
    NEED_CLUSTER = ['ocae']
    def __init__(self, cfg, aug_dict, is_training=True) -> None:
        # self.cfg = cfg
        # self.model_name = cfg.MODEL.name
        # self.aug = aug
        super(VideoAnomalyDatasetFactory, self).__init__(cfg, aug_dict, is_training)
        self.aug_dict = aug_dict
        # import ipdb; ipdb.set_trace()
        self.ingredient = DATASET_REGISTRY.get(self.dataset_name)
        self._jude_need_cluster()
        self._jude_need_w()
    
    def _jude_need_w(self):
        if self.model_name in VideoAnomalyDatasetFactory.NEED_W:
            self.need_w_flag = True
        else:
            self.need_w_flag = False
        # pass

    def _jude_need_cluster(self):
        if self.model_name in VideoAnomalyDatasetFactory.NEED_CLUSTER:
            self.need_cluster_flag = True
        else:
            self.need_cluster_flag = False
        # pass

    def _produce_train_dataset(self):
        train_dataset_dict = OrderedDict()
        train_dataset = self.ingredient(self.dataset_params.train_path, clip_length=self.dataset_params.train_clip_length, 
                                        sampled_clip_length=self.dataset_params.train_sampled_clip_length, 
                                        frame_step=self.dataset_params.train_frame_step,clip_step=self.dataset_params.train_clip_step, is_training=True,
                                        transforms=self.aug_dict['train_aug'], cfg=self.cfg)
        train_dataset_dict['video_keys'] = ['all']
        train_dataset_dict['video_datasets'] = OrderedDict()
        train_dataset_dict['video_datasets']['all'] = train_dataset
        # pass 
        return train_dataset_dict
    
    def _produce_test_dataset(self):
        dataset_dict = OrderedDict()
        video_dirs = os.listdir(self.dataset_params.test_path)
        video_dirs.sort()
        for video_dir in video_dirs:
            _temp_test_folder = os.path.join(self.dataset_params.test_path, video_dir)
            dataset = self.ingredient(_temp_test_folder, clip_length=self.dataset_params.test_clip_length, 
                                      sampled_clip_length=self.dataset_params.test_sampled_clip_length, 
                                      clip_step=self.dataset_params.test_clip_step, frame_step=self.dataset_params.test_frame_step, is_training=False,
                                      transforms=self.aug_dict['test_aug'], one_video=True, cfg=self.cfg)
            dataset_dict[video_dir] = dataset
        video_keys = list(dataset_dict.keys())
        test_dataset_dict = OrderedDict()
        test_dataset_dict['video_keys'] = video_keys
        test_dataset_dict['video_datasets'] = dataset_dict
        return test_dataset_dict
        # pass
    
    def _produce_w_dataset(self):
        dataset_dict = OrderedDict()
        video_dirs = os.listdir(self.dataset_params.train_path)
        video_dirs.sort()
        for video_dir in video_dirs:
            _temp_folder = os.path.join(self.dataset_params.train_path, video_dir)
            # if self.dataset_name == 'shanghai':
            #     dataset = self.ingredient(_temp_test_folder, clip_length=self.dataset_params.test_clip_length, sampled_clip_length=self.dataset_params.test_sampled_clip_length, 
            #                                clip_step=self.dataset_params.test_clip_step, frame_step=self.dataset_params.test_frame_step, one_video=True, mini=True, is_training=True, 
            #                                transforms=self.aug, cfg=self.cfg)
            #     print(f'\033[1;31mUsing the MINI dataset of {cfg.DATASET.name} for calc the w \033[0m')
            # else:
            #     dataset = AvenuePedShanghaiAll(_temp_test_folder, clip_length=cfg.DATASET.train_clip_length, sampled_clip_length=cfg.DATASET.train_sampled_clip_length, 
            #                                clip_step=cfg.DATASET.train_clip_step, frame_step=cfg.DATASET.train_frame_step,transforms=aug, one_video=True, cfg=cfg, phase='train')
            dataset = self.ingredient(_temp_folder, clip_length=self.dataset_params.train_clip_length, sampled_clip_length=self.dataset_params.train_clip_length, 
                                        clip_step=self.dataset_params.train_clip_length, frame_step=self.dataset_params.train_clip_length, one_video=True, is_training=True, 
                                        transforms=self.aug_dict['train_aug'], cfg=self.cfg)
            dataset_dict[video_dir] = dataset
        video_keys = list(dataset_dict.keys())
        # pass
        w_dataset_dict = OrderedDict()
        w_dataset_dict['video_keys'] = video_keys
        w_dataset_dict['video_datasets'] = dataset_dict
        return w_dataset_dict

    def _produce_cluster_dataset(self):
        dataset_dict = OrderedDict()
        video_dirs = os.listdir(self.dataset_params.train_path)
        video_dirs.sort()
        for video_dir in video_dirs:
            _temp_folder = os.path.join(self.dataset_params.train_path, video_dir)
            dataset = self.ingredient(_temp_folder, clip_length=self.dataset_params.train_clip_length, sampled_clip_length=self.dataset_params.train_sampled_clip_length, 
                                       clip_step=self.dataset_params.train_clip_step, frame_step=self.dataset_params.train_frame_step, is_training=True, 
                                       transforms=self.aug_dict['train_aug'], one_video=True, cfg=self.cfg)
            dataset_dict[video_dir] = dataset
        video_keys = list(dataset_dict.keys())
        cluster_dataset_dict = OrderedDict()
        cluster_dataset_dict['video_keys'] = video_keys
        cluster_dataset_dict['video_datasets'] = dataset_dict
        return cluster_dataset_dict
        # pass
    

    def _build(self):
        # if self.model_name in VideoAnomalyDatasetFactory.NORMAL:
        #     dataloader = self._build_normal()
        # elif self.model_name in VideoAnomalyDatasetFactory.CLASS1:
        #     dataloader = self._build_class1()
        # elif self.model_name in VideoAnomalyDatasetFactory.CLASS2:
        #     dataloader = self._build_class2()
        # else:
        #     raise Exception('123')
        # return dataloader
        dataset_dict = OrderedDict()
        test_dataset_dict = self._produce_test_dataset()
        dataset_dict['test_dataset_dict'] = OrderedDict()
        dataset_dict['test_dataset_dict']['general_dataset_dict'] = test_dataset_dict

        if self.is_training:
            train_dataset_dict = self._produce_train_dataset()
            # general_dataset_dict = OrderedDict()
            # general_dataset_dict['general_dataset_dict'] = train_dataset_dict
            dataset_dict['train_dataset_dict'] = OrderedDict()
            dataset_dict['train_dataset_dict']['general_dataset_dict']  = train_dataset_dict
            if self.need_w_flag:
                w_dataset = self._produce_w_dataset()
                # w_dataset_dict = OrderedDict()
                # w_dataset_dict['w_dataset_dict'] = w_dataset
                dataset_dict['train_dataset_dict']['w_dataset_dict'] = w_dataset
            if self.need_cluster_flag:
                cluster_dataset = self._produce_cluster_dataset()
                # cluster_dataset_dict = OrderedDict()
                # cluster_dataset_dict['cluster_dataset_dict'] = cluster_dataset
                dataset_dict['train_dataset_dict']['cluster_dataset_dict'] = cluster_dataset

        return dataset_dict

    def __call__(self):
        dataset_dict = self._build()
        return dataset_dict

# def _get_test_dataset(cfg, aug)->(list, list):
#     dataset_list = OrderedDict()    
#     video_dirs = os.listdir(cfg.DATASET.test_path)
#     video_dirs.sort()
#     for t_dir in video_dirs:
#         _temp_test_folder = os.path.join(cfg.DATASET.test_path, t_dir)
#         dataset = AvenuePedShanghaiAll(_temp_test_folder, clip_length=cfg.DATASET.test_clip_length, sampled_clip_length=cfg.DATASET.test_sampled_clip_length, 
#                                     clip_step=cfg.DATASET.test_clip_step, frame_step=cfg.DATASET.test_frame_step,transforms=aug, one_video=True, cfg=cfg, phase='val')
#         dataset_list[t_dir] = dataset
#     video_keys = list(dataset_list.keys())
#     return (dataset_list, video_keys)

# def _get_train_w_dataset(cfg, aug)->(list, list):
#     dataset_list = OrderedDict()
#     video_dirs = os.listdir(cfg.DATASET.train_path)
#     video_dirs.sort()
#     for t_dir in video_dirs:
#         _temp_test_folder = os.path.join(cfg.DATASET.train_path, t_dir)
#         if cfg.DATASET.name == 'shanghai':
#             dataset = AvenuePedShanghaiAll(_temp_test_folder, clip_length=cfg.DATASET.test_clip_length, sampled_clip_length=cfg.DATASET.test_sampled_clip_length, 
#                                            clip_step=cfg.DATASET.test_clip_step, frame_step=cfg.DATASET.test_frame_step, one_video=True, mini=True, transforms=aug, cfg=cfg, phase='train')
#             print(f'\033[1;31mUsing the MINI dataset of {cfg.DATASET.name} for calc the w \033[0m')
#         else:
#             dataset = AvenuePedShanghaiAll(_temp_test_folder, clip_length=cfg.DATASET.train_clip_length, sampled_clip_length=cfg.DATASET.train_sampled_clip_length, 
#                                            clip_step=cfg.DATASET.train_clip_step, frame_step=cfg.DATASET.train_frame_step,transforms=aug, one_video=True, cfg=cfg, phase='train')
#         dataset_list[t_dir] = dataset
#     video_keys = list(dataset_list.keys())
#     return (dataset_list, video_keys)

# def _get_cluster_dataset(cfg, aug)->(list, list):
#     dataset_list = OrderedDict()
#     video_dirs = os.listdir(cfg.DATASET.train_path)
#     video_dirs.sort()
#     for t_dir in video_dirs:
#         _temp_train_folder = os.path.join(cfg.DATASET.train_path, t_dir)
#         dataset = AvenuePedShanghaiAll(_temp_train_folder, clip_length=cfg.DATASET.train_clip_length, sampled_clip_length=cfg.DATASET.train_sampled_clip_length, 
#                                        clip_step=cfg.DATASET.train_clip_step, frame_step=cfg.DATASET.train_frame_step,is_training=True, transforms=aug, one_video=True, cfg=cfg, phase='train')
#         dataset_list[t_dir] = dataset
#     video_keys = list(dataset_list.keys())
#     return (dataset_list, video_keys)