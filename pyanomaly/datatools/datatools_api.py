"""
@author:  Yuhao Cheng
@contact: yuhao.cheng[at]outlook.com
"""
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict
from .dataclass.sampler import TrainSampler, DistTrainSampler
from .abstract.abstract_datasets_builder import AbstractBuilder
from .dataclass.augment import AugmentAPI
from .datatools_registry import DATASET_FACTORY_REGISTRY, EVAL_METHOD_REGISTRY
from .dataclass import *
from .evaluate import *

import logging
logger = logging.getLogger(__name__)

class DataAPI(AbstractBuilder):
    """
    The API class is used for getting the torch.data.Dataloader dictionary
    """
    NAME = 'DatasetAPI'
    def __init__(self, cfg, is_training):
        """
        The initialization method of DataAPI
        Args:
            cfg: The config object
            is_training: indicate whether dataloaders are using in the training process
        Returns:
            None
        """
        self.seed = cfg.DATASET.seed
        self.cfg = cfg
        self.is_training = is_training
        aug_api = AugmentAPI(cfg)
        aug_dict = aug_api.build()
        self.factory = DATASET_FACTORY_REGISTRY.get(self.cfg.DATASET.factory)(self.cfg, aug_dict, self.is_training)

    def build(self):
        """
        The building method
        Args:
            None
        Returns:
            dataloader_dict(OrderedDict): The dictionary contains the dataloader used in training or val/test process
            For example:
            {
                'train':{
                    'general_data_dict':{
                        'all': torch.data.Dataloader
                    },
                    'w_dataset_dict': {
                        '01': torch.data.Dataloader, 
                        ... ...
                    },
                    ... ...
                },
                'val':{
                    'general_dataset_dict':{
                        '01': torch.data.Dataloader,
                        '02': torch.data.Dataloader,
                        '03': torch.data.Dataloder,
                        ... ...
                    },
                    ... ...
                }
            }
        """
        # build the dataset
        dataset_all = self._build_dataset()
        
        # initialize the dataloader dict
        dataloader_dict = OrderedDict()
        dataloader_dict['train'] = OrderedDict()
        dataloader_dict['val'] = OrderedDict()

        # build the val part of dataloder dict
        dataset_dict = dataset_all['val_dataset_dict']
        batch_size = self.cfg.VAL.batch_size
        
        for key in dataset_dict.keys():
            temp = dataset_dict[key]
            dataloader_dict['val'][key] = OrderedDict()
            for dataset_key in temp['video_keys']:
                dataset = dataset_dict[key]['video_datasets'][dataset_key]
                temp_data_len = len(dataset)
                sampler = self._build_sampler(temp_data_len)
                batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last=True)
                dataloader = DataLoader(dataset, batch_sampler=batch_sampler, pin_memory=True, num_workers=self.cfg.DATASET.num_workers)
                dataloader_dict['val'][key][dataset_key] = dataloader
        
        # build the train part of dataloader dict
        if self.is_training:
            dataset_dict = dataset_all['train_dataset_dict']
            batch_size = self.cfg.TRAIN.batch_size
            for key in dataset_dict.keys():
                temp = dataset_dict[key]
                dataloader_dict['train'][key] = OrderedDict()
                for dataset_key in temp['video_keys']:
                    # import ipdb; ipdb.set_trace()
                    dataset = dataset_dict[key]['video_datasets'][dataset_key]
                    temp_data_len = len(dataset)
                    # need to change
                    sampler = self._build_sampler(temp_data_len)
                    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last=True)
                    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, pin_memory=True)
                    dataloader_dict['train'][key][dataset_key] = dataloader
        
        return dataloader_dict
    
    def _build_dataset(self):
        """
        Using the dataset factory to get the dictionaty of torch.data.Dataset
        Args:
            None
        Returns:
            dataset(OrderedDict)
        """
        dataset = self.factory()
        return dataset
    
    def _build_sampler(self, _data_len):
        """
        The method is used to build the sampler based on the length of the dataset
        Args:
            _data_len: The length of the dataset
        Returns:
            sampler: torch.data.Sampler
        """
        if self.cfg.SYSTEM.distributed.use:
            sampler = DistTrainSampler(_data_len)
        else:
            sampler = TrainSampler(_data_len, self.seed)
        return sampler
    
    def __call__(self):
        """
        The method calls the self.build()
        Args:
            None
        Returns:
            None
        """
        dataloader_dict = self.build()
        return dataloader_dict


class EvaluateAPI(object):
    """
    The class to get the Evaluation object
    """
    def __init__(self, cfg, is_training):
        """
        The initialization method
        Args:
            cfg: The config object
            is_training: indicate whether the evaluation function is used in training process
        Returns:
            None
        """
        self.cfg = cfg
        self.eval_name = cfg.DATASET.evaluate_function.name
        self.is_training = is_training
    
    def __call__(self):
        """
        The method is used for get the evaluation function objects
        Args:
            None
        Returns:
            None
        """

        eval_method = EVAL_METHOD_REGISTRY.get(self.eval_name)(self.cfg, self.is_training)
        logger.info(f'Use the eval method {self.eval_name}')

        return eval_method 
