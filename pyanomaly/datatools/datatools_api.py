import torch
from torch.utils.data import DataLoader
from collections import OrderedDict
from .dataclass.sampler import TrainSampler, DistTrainSampler
from .abstract.abstract_datasets_builder import AbstractBuilder
from .datatools_registry import DATASET_FACTORY_REGISTRY
from .dataclass import *
from .dataclass.augment import AugmentAPI
from .evaluate.eval_function import eval_functions

import logging
logger = logging.getLogger(__name__)

# BUILTIN = ['avenue', 'shanghai', 'vad', 'ped1', 'ped2', 'dota']

class DataAPI(AbstractBuilder):
    _name = 'DatasetAPI'
    def __init__(self, cfg, is_training):
        self.seed = cfg.DATASET.seed
        self.cfg = cfg
        self.is_training = is_training
        aug_api = AugmentAPI(cfg)
        aug_dict = aug_api.build()
        self.factory = DATASET_FACTORY_REGISTRY.get(self.cfg.DATASET.factory)(self.cfg, aug_dict, self.is_training)

    def build(self):
        '''
        flag: the type of the dataset
        train--> use to train, all data, inf sampler
        test--> use to val/test, dataset for each video, no-inf sampler
        '''
        # build the dataset
        dataset_all = self._build_dataset()
        
        dataloader_dict = OrderedDict()
        dataloader_dict['train'] = OrderedDict()
        dataloader_dict['test'] = OrderedDict()

        dataset_dict = dataset_all['test_dataset_dict']
        batch_size = self.cfg.VAL.batch_size
        
        for key in dataset_dict.keys():
            temp = dataset_dict[key]
            dataloader_dict['test'][key] = OrderedDict()
            for dataset_key in temp['video_keys']:
                dataset = dataset_dict[key]['video_datasets'][dataset_key]
                temp_data_len = len(dataset)
                sampler = self._build_sampler(temp_data_len)
                batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last=True)
                dataloader = DataLoader(dataset, batch_sampler=batch_sampler, pin_memory=True, num_workers=self.cfg.DATASET.num_workers)
                dataloader_dict['test'][key][dataset_key] = dataloader
        
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
        
        # import ipdb; ipdb.set_trace()
        return dataloader_dict
    
    def _build_dataset(self):
        
        dataset = self.factory()
        return dataset
    
    def _build_sampler(self, _data_len):
        if self.cfg.SYSTEM.distributed.use:
            sampler = DistTrainSampler(_data_len)
        else:
            sampler = TrainSampler(_data_len, self.seed)
        return sampler
    
    def __call__(self):
        dataloader_dict = self.build()
        return dataloader_dict


class EvaluateAPI(object):
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
    
    def __call__(self, eval_function_type):
        assert eval_function_type in eval_functions, f'there is no type of evaluation {eval_function_type}, please check {eval_functions.keys()}'
        self.logger.info(f'==> Using the eval function: {eval_function_type}')
        t = eval_functions[eval_function_type]
        return t
